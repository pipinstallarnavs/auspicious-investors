"""Detached Cloud Run execution with ADC, bounded requests and durable GCS checkpoints."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import time
import uuid
import zipfile

from .core import atomic_text, canonical, read_jsonl, run_trials, strict_json
from .gemini import GeminiProvider
from .stimuli import build_trials


def estimate_cost(trials, spec):
    """Conservative planning bound, not a Cloud Billing spending limit.

    UTF-8 request bytes plus 1024 overhead tokens bound text input conservatively.
    Reserve the maximum output and every allowed HTTP attempt; ignore caching discounts.
    Cloud retries are disabled, so they cannot multiply this reservation.
    """
    for value in (spec["model_budget_usd"], *spec["prices_usd_per_million"].values()):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Budget and recorded token prices must be positive finite numbers")
    if "pricing_valid_until" in spec and dt.datetime.now(dt.timezone.utc).date() > dt.date.fromisoformat(spec["pricing_valid_until"]):
        raise ValueError("Recorded model pricing has expired; verify prices before starting a new run")
    provider = GeminiProvider(spec["model"], project=spec["project"], **spec["generation"])
    input_tokens = 0
    for trial in trials:
        body = {"contents": [{"role": "user", "parts": [{"text": trial["prompt"]}]}],
                "generationConfig": provider.generation}
        input_tokens += len(json.dumps(body, ensure_ascii=False).encode("utf-8")) + 1024
    attempts = spec["generation"].get("max_attempts", 4)
    output_tokens = len(trials) * spec["generation"]["max_output_tokens"]
    prices = spec["prices_usd_per_million"]
    upper = attempts * (input_tokens * prices["input"] + output_tokens * prices["output"]) / 1_000_000
    return {"trials": len(trials), "max_attempts_per_trial": attempts,
            "conservative_input_tokens_per_attempt": input_tokens,
            "max_output_tokens_per_attempt": output_tokens,
            "model_cost_upper_bound_usd": upper, "model_budget_usd": spec["model_budget_usd"],
            "pricing_source": spec["pricing_source"],
            "note": "Planning bound at the recorded prices; excludes cloud infrastructure and tax."}


class Checkpoint:
    def __init__(self, bucket, prefix, interval=30):
        self.bucket, self.prefix, self.interval = bucket, prefix, interval
        self.last = 0.0

    def __call__(self, connection, directory, force=False):
        if not force and time.monotonic() - self.last < self.interval:
            return
        target = directory / "checkpoint.sqlite"
        with contextlib.closing(sqlite3.connect(target)) as backup:
            connection.backup(backup)
            # A connection context manager commits but does not close. Flush copied
            # WAL pages into one self-contained file before uploading the snapshot.
            backup.execute("PRAGMA journal_mode=DELETE")
        self.bucket.blob(self.prefix + "/checkpoint.sqlite").upload_from_filename(str(target))
        counts = dict(connection.execute("SELECT status,COUNT(*) FROM results GROUP BY status"))
        self.bucket.blob(self.prefix + "/progress.json").upload_from_string(canonical({
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "counts": counts,
            "stored": sum(counts.values()), "state": "running"}), content_type="application/json")
        self.last = time.monotonic()


def upload_tree(bucket, prefix, directory):
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != ".lock" and not path.name.endswith(("-wal", "-shm")):
            bucket.blob(prefix + "/" + path.relative_to(directory).as_posix()).upload_from_filename(str(path))


def run_cloud(bucket_name, run_id, directory="/work"):
    from google.cloud import storage
    from google.api_core.exceptions import PreconditionFailed

    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run_id):
        raise ValueError("Invalid run identifier")
    bucket = storage.Client().bucket(bucket_name)
    prefix = "runs/" + run_id
    raw = bucket.blob(prefix + "/run.json").download_as_bytes()
    spec = strict_json(raw.decode("utf-8"))
    spec_hash = hashlib.sha256(raw).hexdigest()
    status_blob = bucket.blob(prefix + "/status.json")
    if status_blob.exists():
        prior = strict_json(status_blob.download_as_text())
        if prior.get("spec_sha256") != spec_hash:
            raise ValueError("Run specification changed; use a new run identifier")
        if prior.get("state") in {"completed", "completed_with_errors"}:
            print(canonical(prior), flush=True)
            return prior
    owner = os.environ.get("CLOUD_RUN_EXECUTION") or uuid.uuid4().hex
    lease = bucket.blob(prefix + "/lease.json")
    try:
        lease.upload_from_string(canonical({"owner": owner, "created_at": dt.datetime.now(dt.timezone.utc).isoformat()}),
                                 content_type="application/json", if_generation_match=0)
    except PreconditionFailed:
        raise ValueError("This run is already owned by another execution; inspect its lease before resuming") from None
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    state = {"state": "running", "run_id": run_id, "spec_sha256": spec_hash, "execution": owner,
             "project": spec["project"], "model": spec["model"], "bucket": bucket_name,
             "source_revision": spec.get("source_revision"), "container_image": spec.get("container_image"),
             "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    checkpoint_blob = bucket.blob(prefix + "/checkpoint.sqlite")
    try:
        status_blob.upload_from_string(canonical(state), content_type="application/json")
        inputs = root / "inputs"
        inputs.mkdir(exist_ok=True)
        design = Path(spec["design_config"])
        shutil.copyfile(design, inputs / "design.json")
        shutil.copyfile("data/stimuli.json", inputs / "stimuli.json")
        atomic_text(inputs / "run.json", raw.decode("utf-8"))
        built = build_trials(design, inputs / "trials.jsonl")
        trials = read_jsonl(inputs / "trials.jsonl")
        estimate = estimate_cost(trials, spec)
        if estimate["model_cost_upper_bound_usd"] > spec["model_budget_usd"]:
            raise ValueError("Conservative request-cost bound exceeds this run's model budget")
        atomic_text(root / "cost-plan.json", json.dumps(estimate, indent=2) + "\n")
        atomic_text(root / "design-summary.json", json.dumps(built, indent=2) + "\n")
        upload_tree(bucket, prefix, inputs.parent)
        output = root / "model"
        output.mkdir(exist_ok=True)
        if checkpoint_blob.exists():
            checkpoint_blob.download_to_filename(str(output / "journal.sqlite"))
        provider = GeminiProvider(spec["model"], project=spec["project"], **spec["generation"])
        summary = run_trials(inputs / "trials.jsonl", output, provider,
                             concurrency=spec["concurrency"],
                             checkpoint=Checkpoint(bucket, prefix))
        from .analysis import analyze
        analyze(output / "records.jsonl", root / "analysis", bootstrap=spec.get("bootstrap", 2000))
        records = read_jsonl(output / "records.jsonl")
        usage = {"promptTokenCount": 0, "candidatesTokenCount": 0, "thoughtsTokenCount": 0,
                 "totalTokenCount": 0, "cachedContentTokenCount": 0}
        for record in records:
            for key in usage:
                usage[key] += record["result"].get("usage", {}).get(key, 0)
        state.update(state="completed_with_errors" if summary["counts"].get("error") else "completed",
                     summary=summary, usage=usage, cost_plan=estimate,
                     finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in root.rglob("*") if p.is_file() and p.name != ".lock"
                  and not p.name.endswith(("-wal", "-shm"))}
        atomic_text(root / "checksums.json", json.dumps(hashes, indent=2) + "\n")
        with zipfile.ZipFile(root / "results.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file() and path.name not in {"results.zip", ".lock"} and not path.name.endswith(("-wal", "-shm")):
                    archive.write(path, path.relative_to(root))
        upload_tree(bucket, prefix, root)
        status_blob.upload_from_string(canonical(state), content_type="application/json")
        print(canonical(state), flush=True)
        return state
    except BaseException as exc:
        state.update(state="failed", error_type=type(exc).__name__,
                     finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        status_blob.upload_from_string(canonical(state), content_type="application/json")
        raise
    finally:
        lease.delete(if_generation_match=lease.generation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run_cloud(args.bucket, args.run_id)


if __name__ == "__main__":
    main()
