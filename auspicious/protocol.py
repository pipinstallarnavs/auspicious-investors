"""Collection-time bindings and attributable research provenance, without inventing review."""
from __future__ import annotations

import datetime as dt
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
from urllib.parse import urlparse


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_identity():
    root = Path(__file__).resolve().parent
    files = {p.name: sha256(p) for p in sorted(root.glob("*.py"))}
    versions = {}
    for name in ("auspicious-investors", "numpy", "torch", "transformers", "tokenizers", "google-auth"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True,
                                  text=True, check=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        revision = None
    return {"source_sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
            "source_files": files, "git_revision": revision, "python": platform.python_version(),
            "platform": platform.platform(), "dependencies": versions}


def registration(plan):
    entry = plan.get("registration")
    if not isinstance(entry, dict):
        raise ValueError("Prospective inference requires an external registration record")
    url = urlparse(entry.get("url", ""))
    if url.scheme != "https" or not url.netloc:
        raise ValueError("Registration requires an external HTTPS URL")
    try:
        timestamp = dt.datetime.fromisoformat(entry["registered_at"].replace("Z", "+00:00"))
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError("Registration requires a timezone-aware registered_at timestamp") from exc
    if timestamp.tzinfo is None or timestamp >= dt.datetime.now(dt.timezone.utc):
        raise ValueError("Registration must precede collection and include its timezone")
    if not str(entry.get("timestamp_verified_by", "")).strip():
        raise ValueError("External registration timestamp needs an attributable verifier")
    return {**entry, "verification": "researcher_attestation_not_independently_verified_by_software"}


def collection_protocol(path, *, confirmatory=False):
    if path is None:
        if confirmatory:
            raise ValueError("Confirmatory collection requires a prospectively registered --prereg plan")
        return None
    from .core import strict_json
    raw = Path(path).read_bytes()
    plan = strict_json(raw.decode("utf-8"))
    if not isinstance(plan, dict):
        raise ValueError("Protocol must be a JSON object")
    if confirmatory:
        from .analysis import GROUP_FIELDS, _load_prereg
        _load_prereg(path)
        registration(plan)
        if not plan.get("target_population"):
            raise ValueError("Confirmatory protocol must define target_population")
        if plan.get("evaluation_splits") != ["test"]:
            raise ValueError("Confirmatory protocol must freeze evaluation_splits=['test']")
        primary = plan.get("primary", [])
        if not primary or any(type(p.get("expected_tests")) is not int or p["expected_tests"] < 1 for p in primary):
            raise ValueError("Every confirmatory primary entry requires expected_tests")
        for entry in primary:
            if entry["expected_tests"] != 1 or set(entry.get("filters", {})) != {"source", *GROUP_FIELDS}:
                raise ValueError("Register each primary hypothesis identity separately: expected_tests=1 and all source/design/provider filters")
        if plan.get("template_weighting") != "equal_planned_templates_complete_case":
            raise ValueError("Confirmatory protocol must declare equal_planned_templates_complete_case weighting")
    return {"sha256": hashlib.sha256(raw).hexdigest(), "plan": plan, "text": raw.decode("utf-8")}


def immutable_provider(identity):
    if re.fullmatch(r"[0-9a-f]{64}", str(identity.get("local_model_sha256", ""))):
        return True
    return all(re.fullmatch(r"[0-9a-f]{40,64}", str(identity.get(key, "")))
               for key in ("resolved_revision", "tokenizer_resolved_revision"))


def validate_roster(records, trials):
    """Refuse results relabelled away from the supplied original task roster."""
    by_id = {trial["trial_id"]: trial for trial in trials}
    fields = ("pair_id", "cluster_id", "scenario_id", "split", "language", "festival", "cue", "role",
              "persona_id", "template_id", "paraphrase_id", "replicate", "condition", "event_valence",
              "economics", "prompt", "prompt_sha256", "data_sha256", "config_sha256", "calendar_sha256")
    for record in records:
        trial = record.get("trial", {})
        expected = by_id.get(trial.get("trial_id"))
        if expected is None or any(trial.get(f) != expected.get(f) for f in fields):
            raise ValueError("Result task metadata/prompt does not match the original trial roster")
    return by_id


def verify_bound_records(records, plan_sha256, *, trials=None, registered_at=None):
    from .core import digest
    for record in records:
        binding = record.get("collection", {})
        if binding.get("protocol_sha256") != plan_sha256:
            raise ValueError("Inference plan was not bound to every record before collection")
        if not binding.get("runtime", {}).get("source_sha256"):
            raise ValueError("Collection source/runtime identity is unresolved")
        identity = binding.get("provider_identity", {})
        if not binding.get("immutable_provider_verified") or not immutable_provider(identity):
            raise ValueError("Confirmatory provider/model/tokenizer identity is unresolved")
        if not binding.get("confirmatory"):
            raise ValueError("Exploratory data cannot become prospective confirmatory data at analysis")
        if trials is not None:
            expected = digest({"trials": sorted(trials, key=lambda t: t["trial_id"]), "provider": identity,
                               "confirmatory": True, "runtime": binding["runtime"], "protocol_sha256": plan_sha256})
            if expected != binding.get("fingerprint"):
                raise ValueError("Collection fingerprint does not match the original roster/protocol/provider/runtime")
        if registered_at is not None:
            try:
                collected = dt.datetime.fromisoformat(record["result"]["started_at"].replace("Z", "+00:00"))
                registered = dt.datetime.fromisoformat(registered_at.replace("Z", "+00:00"))
                if collected.tzinfo is None or collected <= registered:
                    raise ValueError("collection precedes registration")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError("Every response must have a collection timestamp after registration") from exc


def load_assignments(path, trials_path=None):
    """Verify an exported study manifest and its exact assignment CSV."""
    from .core import strict_json
    path = Path(path)
    raw = path.read_bytes()
    manifest = strict_json(raw.decode("utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 2 or not manifest.get("study_id"):
        raise ValueError("A study-scoped schema-v2 assignment manifest is required")
    if not isinstance(manifest.get("files"), dict) or not isinstance(manifest["files"].get("assignments"), str):
        raise ValueError("Study manifest must identify its assignment file")
    assignments_path = (path.parent / manifest["files"]["assignments"]).resolve()
    if not assignments_path.is_relative_to(path.parent.resolve()):
        raise ValueError("Assignment file escapes the study directory")
    if sha256(assignments_path) != manifest.get("assignments_sha256"):
        raise ValueError("Assignment manifest/content hash mismatch")
    if trials_path is not None and sha256(trials_path) != manifest["trials"]["sha256"]:
        raise ValueError("Assignment corpus hash does not match the trial roster")
    with assignments_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {}
    positions = set()
    for row in rows:
        required = ("subject_id", "trial_id", "study_id", "language", "condition", "position", "prompt_sha256", "assigned_at")
        if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
            raise ValueError("Assignment row lacks required provenance fields")
        if (not row["subject_id"].startswith(manifest["study_id"] + "-") or not row["position"].isdigit()
                or int(row["position"]) < 1 or (row["subject_id"], row["position"]) in positions):
            raise ValueError("Assignment subject namespace/position is invalid or duplicated")
        positions.add((row["subject_id"], row["position"]))
        key = (row["subject_id"], row["trial_id"])
        if key in index or row["study_id"] != manifest["study_id"]:
            raise ValueError("Duplicate or mixed-study assignment")
        index[key] = row
    if trials_path is not None:
        from .core import read_jsonl
        trials = {t["trial_id"]: t for t in read_jsonl(trials_path)}
        for row in rows:
            trial = trials.get(row["trial_id"])
            if trial is None or any(row[k] != str(trial[k]) for k in ("language", "condition", "prompt_sha256", "pair_id", "cluster_id")):
                raise ValueError("Assignment task metadata does not match its original roster")
    return {"manifest": manifest, "rows": rows, "index": index,
            "manifest_sha256": hashlib.sha256(raw).hexdigest(), "study_id": manifest["study_id"]}


def assignment_flow(study, records, analysed_ids, trial_ids=None):
    """Assigned -> submitted -> valid -> analysed, retaining true nonresponse."""
    submitted = {}
    for record in records:
        result, trial = record["result"], record["trial"]
        key = (result.get("subject_id"), trial.get("trial_id"))
        assigned = study["index"].get(key)
        provenance = result.get("assignment", {})
        if assigned is None or provenance.get("manifest_sha256") != study["manifest_sha256"]:
            raise ValueError("Human result is not bound to this study's original assignments")
        if any(provenance.get(k) != v for k, v in assigned.items()):
            raise ValueError("Human assignment metadata was altered after import")
        submitted[key] = result
    groups = {}
    for key, assigned in study["index"].items():
        if trial_ids is not None and assigned["trial_id"] not in trial_ids:
            continue
        group = (assigned["study_id"], assigned["language"], assigned["condition"])
        counts = groups.setdefault(group, dict(assigned=0, submitted=0, valid=0, analysed=0))
        counts["assigned"] += 1
        result = submitted.get(key)
        if result is not None:
            counts["submitted"] += 1
            counts["valid"] += result.get("status") == "ok"
            counts["analysed"] += key in analysed_ids
    return [{"study_id": key[0], "language": key[1], "condition": key[2], **counts,
             "nonresponse": counts["assigned"] - counts["submitted"]}
            for key, counts in sorted(groups.items())]


def administration_summary(records):
    ordered = {}
    missing = 0
    for record in records:
        result = record["result"]
        if not result.get("submitted_at"):
            missing += 1
            continue
        timestamp = dt.datetime.fromisoformat(result["submitted_at"].replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Response timestamps must include a timezone")
        assigned = result.get("assignment", {})
        started = dt.datetime.fromisoformat(assigned["assigned_at"].replace("Z", "+00:00"))
        if timestamp < started:
            raise ValueError("Response precedes its assignment")
        ordered.setdefault(result["subject_id"], []).append((int(assigned["position"]), timestamp))
    violations = [subject for subject, values in ordered.items()
                  if any(a[1] > b[1] for a, b in zip(sorted(values), sorted(values)[1:]))]
    return {"n_responses": len(records), "responses_without_timestamps": missing, "subjects_with_order_violations": sorted(violations),
            "order_verified": bool(records) and missing == 0 and not violations,
            "limitation": "Timestamps alone cannot verify attention, comprehension, masking, or honest administration."}
