"""Strict response contracts and a single-writer, resumable experiment journal."""
from __future__ import annotations

import concurrent.futures
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import time

PAYLOAD_KEYS = {
    "gold_pct", "equity_pct", "cash_pct", "buy_now", "timing_days",
    "willingness_to_pay_inr", "expected_return_pct", "rationale",
}
RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "gold_pct": {"type": "NUMBER"},
        "equity_pct": {"type": "NUMBER"},
        "cash_pct": {"type": "NUMBER"},
        "buy_now": {"type": "BOOLEAN"},
        "timing_days": {"type": "INTEGER"},
        "willingness_to_pay_inr": {"type": "NUMBER"},
        "expected_return_pct": {"type": "NUMBER"},
        "rationale": {"type": "STRING"},
    },
    "required": sorted(PAYLOAD_KEYS),
}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON constant: {value}")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text):
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique_keys)


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for n, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = strict_json(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected JSON object")
                rows.append(row)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
    return rows


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path, rows):
    atomic_text(path, "".join(canonical(row) + "\n" for row in rows))


def load_records(path):
    return read_jsonl(path)


CONDITIONS = ("event", "neutral")
CONTROL_CONDITIONS = ("no_calendar", "occasion_control", "purchase_control")


def load_review_context(trials_path, confirmatory):
    """Review materials live in a per-corpus sidecar; only confirmatory work needs them."""
    if not confirmatory:
        return None
    from .stimuli import load_review_context as load
    return load(trials_path)


def validate_trials(trials, confirmatory=False, review_context=None):
    if not trials:
        raise ValueError("No trials")
    ids, pairs, splits = set(), {}, {}
    required = {
        "trial_id", "pair_id", "cluster_id", "split", "language", "festival",
        "condition", "event_valence", "cue", "role", "persona_id", "scenario_id",
        "paraphrase_id", "template_id", "replicate", "prompt", "prompt_sha256", "review_status", "review",
    }
    for trial in trials:
        missing = required - trial.keys()
        if missing:
            raise ValueError(f"Trial missing fields: {sorted(missing)}")
        if trial["trial_id"] in ids:
            raise ValueError(f"Duplicate trial_id: {trial['trial_id']}")
        ids.add(trial["trial_id"])
        if not isinstance(trial["prompt"], str) or not trial["prompt"].strip():
            raise ValueError("Empty/non-text prompt")
        if hashlib.sha256(trial["prompt"].encode()).hexdigest() != trial["prompt_sha256"]:
            raise ValueError("Prompt content hash mismatch; rebuild trials")
        for field, allowed in {
            "condition": {*CONDITIONS, *CONTROL_CONDITIONS}, "split": {"train", "dev", "test"},
            "language": {"en", "hi", "ta", "gu", "bn"},
            "festival": {"dhanteras", "akshaya_tritiya", "muhurat", "pitru_paksha"},
            "event_valence": {"auspicious", "avoidance"},
            "cue": {"date_only", "festival_named", "explicit_belief"},
            "role": {"investor", "advisor"}, "review_status": {"draft", "approved"},
        }.items():
            if trial[field] not in allowed:
                raise ValueError(f"Invalid {field}: {trial[field]}")
        if type(trial["replicate"]) is not int or trial["replicate"] < 0:
            raise ValueError("replicate must be a nonnegative integer")
        cluster = trial["cluster_id"]
        if cluster in splits and splits[cluster] != trial["split"]:
            raise ValueError(f"Cluster {cluster} leaks across splits")
        splits[cluster] = trial["split"]
        pair = pairs.setdefault(trial["pair_id"], {})
        if trial["condition"] in pair:
            raise ValueError("Duplicate condition within pair")
        pair[trial["condition"]] = trial
        if confirmatory:
            from .stimuli import validate_trial_review
            validate_trial_review(trial, review_context)
    # All factors must match inside a counterfactual pair, except treatment text/review.
    matched_fields = required - {"trial_id", "condition", "prompt", "prompt_sha256", "review_status", "review"}
    condition_sets = set()
    for pair_id, pair in pairs.items():
        if not set(CONDITIONS) <= set(pair):
            raise ValueError(f"Incomplete counterfactual pair: {pair_id}")
        condition_sets.add(frozenset(pair))
        for condition, trial in pair.items():
            for field in matched_fields:
                if pair["event"][field] != trial[field]:
                    raise ValueError(f"Pair {pair_id} mismatches {field} in {condition}")
    if len(condition_sets) > 1:
        raise ValueError("Pairs have inconsistent condition sets; every pair must carry the same controls")
    if confirmatory:
        cohort_hash = digest(sorted(ids))
        if any(trial.get("design_trial_count") != len(trials) or
               trial.get("design_trial_ids_sha256") != cohort_hash for trial in trials):
            raise ValueError("Confirmatory corpus differs from the full reviewed design; rebuild without subsetting")


def validate_payload(payload):
    if not isinstance(payload, dict) or set(payload) != PAYLOAD_KEYS:
        raise ValueError("Response must contain exactly the specified eight fields")
    for key in PAYLOAD_KEYS - {"buy_now", "timing_days", "rationale"}:
        value = payload[key]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{key} must be a finite number")
    for key in ("gold_pct", "equity_pct", "cash_pct"):
        if not 0 <= payload[key] <= 100:
            raise ValueError(f"{key} outside [0,100]")
    if abs(sum(payload[k] for k in ("gold_pct", "equity_pct", "cash_pct")) - 100) > 0.010000001:
        raise ValueError("Allocation percentages must sum to 100")
    if type(payload["buy_now"]) is not bool:
        raise ValueError("buy_now must be boolean")
    if type(payload["timing_days"]) is not int or not 0 <= payload["timing_days"] <= 365:
        raise ValueError("timing_days must be an integer in [0,365]")
    if payload["buy_now"] != (payload["timing_days"] == 0):
        raise ValueError("buy_now must agree with timing_days == 0")
    if payload["willingness_to_pay_inr"] < 0:
        raise ValueError("willingness_to_pay_inr must be nonnegative")
    if not isinstance(payload["rationale"], str) or not payload["rationale"].strip():
        raise ValueError("rationale must be nonempty text")


def normalize_result(result, trial=None):
    result = dict(result)
    result.setdefault("error", None)
    result.setdefault("raw_text", "")
    result.setdefault("payload", None)
    if result.get("status") in {"blocked", "error"}:
        result["payload"] = None
        return result
    try:
        payload = strict_json(result["raw_text"])
        validate_payload(payload)
        if trial is not None:
            max_delay = trial.get("economics", {}).get("max_delay_days", 365)
            if payload["timing_days"] > max_delay:
                raise ValueError("timing_days exceeds this scenario's permitted delay")
        result.update(status="ok", payload=payload, error=None)
    except (ValueError, TypeError, KeyError) as exc:
        result.update(status="invalid", payload=None, error=str(exc))
    return result


def _select_trials(trials, limit):
    pairs = {}
    for trial in trials:
        pairs.setdefault(trial["pair_id"], []).append(trial)
    # Hash order spreads conditions/languages over time without mutable RNG state.
    order = sorted(pairs, key=lambda p: hashlib.sha256(p.encode()).hexdigest())
    if limit is not None:
        size = len(pairs[order[0]])
        if limit < size:
            raise ValueError(f"limit must be >= {size}: complete pairs (all conditions) are retained")
        order = order[:limit // size]
    selected = [t for p in order for t in pairs[p]]
    return sorted(selected, key=lambda t: hashlib.sha256(t["trial_id"].encode()).hexdigest())


def run_trials(trials_path, output_dir, provider, *, confirmatory=False, concurrency=2,
               limit=None, retry_errors=False, checkpoint=None):
    trials = read_jsonl(trials_path)
    validate_trials(trials, confirmatory=confirmatory,
                    review_context=load_review_context(trials_path, confirmatory))
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    if confirmatory and limit is not None:
        raise ValueError("Confirmatory runs cannot use --limit; predefine the full design")
    selected = _select_trials(trials, limit)
    if hasattr(provider, "prepare"):
        provider.prepare()
    identity = provider.identity
    fingerprint = digest({"trials": sorted(trials, key=lambda t: t["trial_id"]),
                          "provider": identity, "confirmatory": confirmatory})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with open(directory / ".lock", "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process owns this run directory") from exc
        with contextlib.closing(sqlite3.connect(directory / "journal.sqlite")) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS results (trial_id TEXT PRIMARY KEY, status TEXT NOT NULL, record TEXT NOT NULL)")
            previous = connection.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()
            if previous and previous[0] != fingerprint:
                raise ValueError("Run fingerprint changed: use a new output directory (prompts/model/settings/review changed)")
            if not previous:
                connection.execute("INSERT INTO metadata VALUES ('fingerprint', ?)", (fingerprint,))
                manifest = {"schema_version": 1, "fingerprint": fingerprint, "provider": identity,
                            "confirmatory": confirmatory, "trial_count": len(trials),
                            "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                            "design_sha256": digest(sorted(trials, key=lambda t: t["trial_id"])),
                            "missingness_policy": "Preserve invalid/blocked/error; never impute as zero",
                            "resume_policy": "Completed results not regenerated; --retry-errors retries transport errors only"}
                connection.execute("INSERT INTO metadata VALUES ('manifest', ?)", (canonical(manifest),))
                connection.commit()
            manifest = strict_json(connection.execute("SELECT value FROM metadata WHERE key='manifest'").fetchone()[0])
            atomic_text(directory / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
            completed = dict(connection.execute("SELECT trial_id,status FROM results"))
            pending = [t for t in selected if t["trial_id"] not in completed or
                       (retry_errors and completed[t["trial_id"]] == "error")]

            def execute(trial):
                started = time.monotonic()
                timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                try:
                    generated = (provider.generate_trial(trial) if hasattr(provider, "generate_trial")
                                 else provider.generate(trial["prompt"]))
                    result = normalize_result(generated, trial=trial)
                except Exception as exc:
                    # Exception text may contain credentials or a request URL. Do not serialize it.
                    result = {"status": "error", "raw_text": "", "payload": None,
                              "error": f"Provider exception: {type(exc).__name__}", "attempts": 1}
                for field in ("provider", "requested_model", "generation"):
                    result.setdefault(field, identity.get(field))
                result.setdefault("model_version", None)
                result.setdefault("usage", {})
                result.setdefault("attempts", 1)
                result.update(started_at=timestamp, elapsed_seconds=time.monotonic() - started)
                return {"trial": trial, "result": result}

            executed = 0
            if checkpoint is not None:
                checkpoint(connection, directory, force=True)
            try:
                # Bound in-flight jobs, rather than enqueueing an entire factorial grid.
                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                    iterator = iter(pending)
                    futures = {}
                    for _ in range(min(concurrency, len(pending))):
                        trial = next(iterator)
                        futures[executor.submit(execute, trial)] = trial["trial_id"]
                    while futures:
                        done, _ = concurrent.futures.wait(futures, return_when=concurrent.futures.FIRST_COMPLETED)
                        for future in done:
                            futures.pop(future)
                            record = future.result()
                            connection.execute("INSERT OR REPLACE INTO results VALUES (?,?,?)",
                                               (record["trial"]["trial_id"], record["result"]["status"], canonical(record)))
                            connection.commit()
                            executed += 1
                            if checkpoint is not None:
                                checkpoint(connection, directory)
                            trial = next(iterator, None)
                            if trial is not None:
                                futures[executor.submit(execute, trial)] = trial["trial_id"]
            finally:
                rows = [strict_json(row[0]) for row in connection.execute("SELECT record FROM results ORDER BY trial_id")]
                write_jsonl(directory / "records.jsonl", rows)
                if checkpoint is not None:
                    checkpoint(connection, directory, force=True)
            counts = {}
            for row in rows:
                status = row["result"]["status"]
                counts[status] = counts.get(status, 0) + 1
            summary = {"executed": executed, "selected": len(selected), "stored": len(rows),
                       "total_trials": len(trials), "counts": counts, "output": str(directory),
                       "fingerprint": fingerprint,
                       "complete": len(rows) == len(trials)}
            atomic_text(directory / "summary.json", json.dumps(summary, indent=2) + "\n")
            return summary
