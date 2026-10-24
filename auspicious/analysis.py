"""Scenario-clustered calendar-cue sensitivity analyses; contextual series are descriptive only.

Estimates describe how a particular model's (or human sample's) answers change when
specified prompt text changes. They are NOT measurements of beliefs or of causal
mediation. Contrasts are A-minus-B over matched conditions of the same pair; repeated
prompts are averaged within scenario, never counted as independent observations. No
outcome is imputed in the main tables; the bounds table shows how much missing answers
could matter.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from statistics import fmean, median

from .core import load_records, normalize_result, read_jsonl, strict_json, validate_trials, write_jsonl
from .inference import ASSUMPTION, MIN_SCENARIOS, estimate, inference_scope

PAYLOAD_METRICS = (
    "gold_pct", "equity_pct", "cash_pct", "buy_now", "timing_days",
    "willingness_to_pay_inr", "expected_return_pct",
)
DERIVED_METRICS = (
    "return_claim_deviation_pp", "return_claim_absolute_error_pp", "return_claim_violation",
    "reserve_shortfall_pct", "reserve_violation", "wtp_premium_pct",
)
METRICS = (*PAYLOAD_METRICS, *DERIVED_METRICS)
RETURN_METRICS = ("return_claim_deviation_pp", "return_claim_absolute_error_pp", "return_claim_violation")
DEFAULT_RETURN_TOLERANCE_PP = 0.01
# Subtracting the same supplied return on both sides leaves the original return contrast.
REDUNDANT_CONTRAST_METRICS = frozenset({"return_claim_deviation_pp"})
# cash_pct is 100 - gold - equity by construction: reported, never a separate finding.
DERIVED_FROM_ALLOCATION = frozenset({"cash_pct"})
UNITS = {
    "gold_pct": "percentage_points", "equity_pct": "percentage_points",
    "cash_pct": "percentage_points", "buy_now": "probability_difference",
    "timing_days": "days", "willingness_to_pay_inr": "INR",
    "expected_return_pct": "percentage_points_annual_expected_return",
    "return_claim_deviation_pp": "percentage_points_annual_expected_return",
    "return_claim_absolute_error_pp": "percentage_points_annual_expected_return",
    "return_claim_violation": "probability_difference",
    "reserve_shortfall_pct": "percentage_points_of_capital",
    "reserve_violation": "probability_difference",
    "wtp_premium_pct": "percent_of_gold_unit_price",
}
PAYLOAD_FIELDS = (*PAYLOAD_METRICS, "rationale")
HUMAN_FIELDS = ("trial_id", "subject_id", *PAYLOAD_FIELDS)
CONTEXT_FIELDS = ("date", "series", "value", "unit", "source", "event")
GROUP_FIELDS = (
    "provider", "requested_model", "model_version", "generation_settings",
    "language", "festival", "cue", "role", "persona_id", "split",
)
TASK_FIELDS = (
    "cluster_id", "scenario_id", "split", "festival", "cue", "role",
    "persona_id", "paraphrase_id", "template_id", "replicate", "event_valence",
)
PAIR_FIELDS = (*TASK_FIELDS, "language")
OPTIONAL_TASK_FIELDS = ("calendar_pair_id",)
META_FIELDS = (*PAIR_FIELDS, *OPTIONAL_TASK_FIELDS, "economics")
STATUSES = ("ok", "invalid", "blocked", "error")
CONDITIONS = ("event", "neutral", "no_calendar", "occasion_control", "purchase_control")
CONTRASTS = {
    "event_minus_neutral": ("event", "neutral"),
    "event_minus_no_calendar": ("event", "no_calendar"),
    "event_minus_occasion_control": ("event", "occasion_control"),
    "event_minus_purchase_control": ("event", "purchase_control"),
    "neutral_minus_no_calendar": ("neutral", "no_calendar"),
}
PRIMARY_CONTRAST = "event_minus_neutral"
# (trial/group field, label, reference level, table name)
INTERACTIONS = (
    ("language", "language", "en", "language_interactions"),
    ("persona_id", "persona", "nonobservant", "persona_interactions"),
    ("cue", "cue", "festival_named", "cue_interactions"),
)
MULTIPLICITY_TABLES = tuple(spec[3] for spec in INTERACTIONS)
FAMILY_TABLES = ("effects", *MULTIPLICITY_TABLES)
HUMAN_DESIGNS = ("auto", "within", "unpaired")
ECONOMICS_NUMBERS = ("capital_inr", "gold_unit_price_inr", "liquidity_need_inr",
                     "gold_expected_annual_pct", "max_delay_days")
# Support of each bounded metric for worst-case (Manski-type) imputation.
BOUNDED = {
    "gold_pct": (0.0, 100.0), "equity_pct": (0.0, 100.0), "cash_pct": (0.0, 100.0),
    "buy_now": (0.0, 1.0), "reserve_violation": (0.0, 1.0), "return_claim_violation": (0.0, 1.0),
}
MIN_SCENARIOS_FOR_GENERAL_CLAIMS = MIN_SCENARIOS
PARTICIPANT_REQUIRED = ("subject_id", "survey_language", "language_proficiency",
                        "financial_literacy", "observance_self_report")
PARTICIPANT_OPTIONAL = ("age_band", "region")
PARTICIPANT_LANGUAGES = ("en", "hi", "ta", "gu", "bn")
VALIDITY_DIMENSIONS = (("all", None), ("language", 4), ("festival", 5), ("cue", 6),
                       ("role", 7), ("persona", 8), ("condition", "condition"))
PREREG_KEYS = {"schema_version", "primary", "primary_adjustment", "exploratory_adjustment", "_note",
               "return_tolerance_pp", "inference_method", "inference_assumption", "registration",
               "evaluation_splits", "target_population", "template_weighting"}
PREREG_ENTRY_KEYS = {"id", "table", "metric", "contrast", "filters", "minimum_meaningful_effect",
                     "expected_tests"}
PREREG_FILTER_KEYS = (*GROUP_FIELDS, "source")
EXPLORATORY_ADJUSTMENTS = ("benjamini_hochberg", "holm", "none")
INFERENCE_FIELDS = ("p_value", "family", "primary_id", "p_adjusted", "adjustment", "n_family",
                    "minimum_meaningful_effect", "exceeds_mme", "within_mme", "derived_from_allocation")


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))


def _csv_write(path, rows, fields):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv_read(path, fields, allowed=None):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"{path}: missing or duplicate CSV headers")
        missing = set(fields) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        if allowed is not None and set(reader.fieldnames) - set(allowed):
            raise ValueError(f"{path}: unknown CSV columns: {', '.join(sorted(set(reader.fieldnames) - set(allowed)))}")
        rows = []
        for line, row in enumerate(reader, 2):
            if None in row or any(row[field] is None for field in fields):
                raise ValueError(f"{path}:{line}: malformed CSV row")
            rows.append((line, row))
        return rows


def _normalize_payload(result, trial):
    """Validate stored payloads too, rather than trusting an 'ok' label."""
    if result.get("status") not in STATUSES:
        return {**result, "status": "invalid", "payload": None,
                "error": "Unknown or missing result status"}
    if result["status"] != "ok":
        return result
    candidate = dict(result)
    try:
        if isinstance(candidate.get("payload"), dict):
            candidate["raw_text"] = _canonical(candidate["payload"])
        return normalize_result(candidate, trial=trial)
    except (ValueError, TypeError, OverflowError) as exc:
        return {**result, "status": "invalid", "payload": None, "error": str(exc)}


def _economics(trial):
    economics = trial.get("economics")
    if not isinstance(economics, dict):
        raise ValueError("missing or invalid trial economics")
    for key in ECONOMICS_NUMBERS:
        value = economics.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"missing or invalid trial economics.{key}")
    if economics["capital_inr"] <= 0 or economics["gold_unit_price_inr"] <= 0:
        raise ValueError("economics capital_inr and gold_unit_price_inr must be positive")
    if economics["liquidity_need_inr"] < 0 or economics["max_delay_days"] < 0:
        raise ValueError("economics liquidity_need_inr and max_delay_days must be nonnegative")
    return economics


def _return_tolerance(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("return_tolerance_pp must be a finite nonnegative number")
    return float(value)


def _metrics(payload, economics, return_tolerance_pp=DEFAULT_RETURN_TOLERANCE_PP):
    """Reported decisions and financial errors, all expressed as floats."""
    values = {name: float(payload[name]) for name in PAYLOAD_METRICS}
    shortfall = max(0.0, 100.0 * economics["liquidity_need_inr"] / economics["capital_inr"]
                    - values["cash_pct"])
    price = economics["gold_unit_price_inr"]
    values["return_claim_deviation_pp"] = values["expected_return_pct"] - economics["gold_expected_annual_pct"]
    error = abs(values["return_claim_deviation_pp"])
    values["return_claim_absolute_error_pp"] = error
    values["return_claim_violation"] = float(error > return_tolerance_pp and not math.isclose(
        error, return_tolerance_pp, rel_tol=1e-12, abs_tol=1e-12))
    values["reserve_shortfall_pct"] = shortfall
    values["reserve_violation"] = 1.0 if shortfall > 0.01 else 0.0
    values["wtp_premium_pct"] = 100.0 * (values["willingness_to_pay_inr"] - price) / price
    return values


def _group(trial, result):
    generation = result.get("generation", {})
    if not isinstance(generation, dict):
        raise ValueError("generation must be an object")
    for name in ("provider", "requested_model"):
        if not isinstance(result.get(name), str) or not result[name]:
            raise ValueError(f"missing {name}")
    version = result.get("model_version")
    if version is not None and (not isinstance(version, str) or not version):
        raise ValueError("model_version must be a nonempty string or null")
    for field in GROUP_FIELDS[4:]:
        if not isinstance(trial.get(field), str) or not trial[field]:
            raise ValueError(f"missing or invalid trial {field}")
    return (result["provider"], result["requested_model"], version,
            _canonical(generation), *(trial[field] for field in GROUP_FIELDS[4:]))


def _group_row(key):
    return dict(zip(GROUP_FIELDS, key))


def _percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _rng(seed, key):
    digest = hashlib.sha256(_canonical([seed, key]).encode()).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _interval(draws):
    return (_percentile(draws, .025), _percentile(draws, .975)) if draws else (None, None)


def _cluster_values(pairs, metric, field="diff"):
    # Replicates within template, templates within subject, subjects within scenario.
    cells = defaultdict(list)
    for pair in pairs:
        reference = pair.get("a") or pair.get("b") or {}
        trial = reference.get("trial", reference)
        cells[(pair["cluster"], pair["subject"], trial.get("template_id"))].append(pair[field][metric])
    subjects = defaultdict(list)
    for (cluster, subject, _template), values in cells.items():
        subjects[cluster, subject].append(fmean(values))
    clusters = defaultdict(list)
    for (cluster, _subject), values in subjects.items():
        clusters[cluster].append(fmean(values))
    return {cluster: fmean(values) for cluster, values in clusters.items()}


def _estimate_all(values_by_metric, bootstrap, rng, *, inferential=True):
    """Default descriptive estimates; optional justified sign-flip p-values, no bootstrap p-values."""
    return {name: estimate([values[k] for k in sorted(values)], inferential=inferential)
            for name, values in values_by_metric.items()}


def _ci_reason(low, bootstrap):
    return "" if low is not None else "population_interval_not_established"


def _inference_blank(metric):
    return {"p_value": None, "family": None, "primary_id": None, "p_adjusted": None,
            "adjustment": None, "n_family": None, "minimum_meaningful_effect": None,
            "exceeds_mme": None, "within_mme": None,
            "derived_from_allocation": metric in DERIVED_FROM_ALLOCATION}


def _task_key(pair, omit=None):
    return (pair["subject"], *(pair["a"].get(field) for field in
                              (*TASK_FIELDS, *OPTIONAL_TASK_FIELDS) if field != omit))


# ---------------------------------------------------------------------------
# Record parsing and pair assembly
# ---------------------------------------------------------------------------

def _prepare(records, label, return_tolerance_pp=DEFAULT_RETURN_TOLERANCE_PP):
    """Parse records into candidates and structurally valid pair buckets.

    A bucket is all records of one (stratum, subject, pair_id). Buckets that are
    duplicated, internally inconsistent or semantically duplicated are rejected
    explicitly and never contribute to any contrast.
    """
    counts = defaultdict(Counter)
    exclusions = []
    warnings = set()
    duplicates = defaultdict(list)
    candidates = []
    rejected_pairs = set()
    seen_pairs = defaultdict(set)
    for index, record in enumerate(records, 1):
        trial = record.get("trial") if isinstance(record, dict) else None
        result = record.get("result") if isinstance(record, dict) else None
        group = None
        pair_key = None
        try:
            if not isinstance(trial, dict) or not isinstance(result, dict):
                raise ValueError("record requires trial and result objects")
            group = _group(trial, result)
            subject = result.get("subject_id")
            if group[0] == "human" and (not isinstance(subject, str) or not subject.strip()):
                raise ValueError("human record requires subject_id")
            if subject is not None and (not isinstance(subject, str) or not subject.strip()):
                raise ValueError("subject_id must be a nonempty string")
            if group[0] != "human" and subject is not None:
                raise ValueError("subject_id is reserved for human observations")
            if not isinstance(trial.get("pair_id"), str) or not trial["pair_id"]:
                raise ValueError("missing or invalid trial pair_id")
            pair_key = (group, subject, trial["pair_id"])
            seen_pairs[group].add((subject, trial["pair_id"]))
            for field in (*PAIR_FIELDS, "pair_id", "trial_id"):
                value = trial.get(field)
                if field == "replicate":
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError("replicate must be a nonnegative integer")
                elif not isinstance(value, str) or not value:
                    raise ValueError(f"missing or invalid trial {field}")
            for field in OPTIONAL_TASK_FIELDS:
                value = trial.get(field)
                if value is not None and (not isinstance(value, str) or not value):
                    raise ValueError(f"invalid trial {field}")
            if trial.get("condition") not in CONDITIONS:
                raise ValueError(f"condition must be one of {', '.join(CONDITIONS)}")
            economics = _economics(trial)
            if (not isinstance(trial.get("prompt"), str) or
                    hashlib.sha256(trial["prompt"].encode()).hexdigest() != trial.get("prompt_sha256")):
                raise ValueError("prompt_hash_mismatch")
            normalized = _normalize_payload(result, trial)
            counts[group][normalized["status"]] += 1
            if group[2] is None:
                warnings.add("Unresolved model_version is kept in its own stratum; cross-run checkpoint identity cannot be established for that stratum.")
            metrics = (_metrics(normalized["payload"], economics, return_tolerance_pp)
                       if normalized["status"] == "ok" else None)
            item = {"trial": trial, "result": normalized, "group": group, "metrics": metrics,
                    "subject": subject, "index": index, "pair_key": pair_key}
            candidates.append(item)
            duplicates[(group[:4], subject, trial["trial_id"])].append(item)
        except (ValueError, TypeError, KeyError) as exc:
            exclusions.append({"source": label, "record_index": index,
                               "pair_id": trial.get("pair_id", "") if isinstance(trial, dict) else "",
                               "contrast": "", "reason": f"malformed_record: {exc}", "detail": ""})
            if pair_key is not None:
                rejected_pairs.add(pair_key)
            if group is not None:
                counts[group]["malformed"] += 1
    for items in duplicates.values():
        if len(items) > 1:
            for item in items:
                rejected_pairs.add(item["pair_key"])
            exclusions.append({"source": label, "record_index": ",".join(str(item["index"]) for item in items),
                               "pair_id": items[0]["trial"]["pair_id"], "contrast": "",
                               "reason": "duplicate_trial_observation: all affected pairs excluded", "detail": ""})
    nominal = defaultdict(set)
    by_pair = defaultdict(list)
    for item in candidates:
        nominal[(item["group"][:2], item["subject"], item["trial"]["pair_id"])].add(item["group"])
        by_pair[item["pair_key"]].append(item)
    for groups in nominal.values():
        if len(groups) > 1:
            # Complete pairs in separate versions are valid separate experiments;
            # incomplete cross-version fragments remain incomplete, never combined.
            warnings.add("A pair_id occurs in multiple model/design strata; strata are never pooled or paired across versions/settings.")
    buckets = []
    for pair_key, items in by_pair.items():
        group, subject, pair_id = pair_key
        reason = None
        conditions = Counter(item["trial"]["condition"] for item in items)
        if pair_key in rejected_pairs:
            reason = "duplicate_or_malformed_record_in_pair"
        elif any(n > 1 for n in conditions.values()):
            reason = "duplicate_condition_in_pair"
        elif any(items[0]["trial"].get(field) != item["trial"].get(field)
                 for item in items[1:] for field in META_FIELDS):
            reason = "inconsistent_pair_metadata"
        bucket = {"group": group, "subject": subject, "pair_id": pair_id, "rejected": reason is not None,
                  "items": {item["trial"]["condition"]: item for item in items}, "records": items}
        if reason:
            exclusions.append({"source": label, "record_index": ",".join(str(item["index"]) for item in items),
                               "pair_id": pair_id, "contrast": "", "reason": reason, "detail": ""})
        buckets.append(bucket)
    # Re-labelling a task/pair must not make a duplicate observation independent.
    semantic = defaultdict(list)
    for bucket in buckets:
        if not bucket["rejected"]:
            first = bucket["records"][0]["trial"]
            semantic[(bucket["group"], bucket["subject"],
                      *(first.get(field) for field in (*TASK_FIELDS, *OPTIONAL_TASK_FIELDS)))].append(bucket)
    for key in sorted((key for key, values in semantic.items() if len(values) > 1), key=_canonical):
        for bucket in semantic[key]:
            bucket["rejected"] = True
        exclusions.append({"source": label, "record_index": "",
                           "pair_id": ",".join(bucket["pair_id"] for bucket in semantic[key]), "contrast": "",
                           "reason": "duplicate_semantic_task: all affected pairs excluded", "detail": ""})
    missingness = []
    for group in sorted(counts, key=_canonical):
        total = sum(counts[group].values())
        for status in (*STATUSES, "malformed"):
            n = counts[group][status]
            missingness.append({**_group_row(group), "source": label, "status": status,
                                "count": n, "denominator": total,
                                "fraction": n / total if total else None})
    if exclusions:
        warnings.add(f"{label}: {len(exclusions)} explicit exclusion entries; inspect exclusions.csv. No missing or invalid outcome was replaced with zero.")
    return {"label": label, "candidates": candidates, "buckets": buckets, "seen": seen_pairs,
            "missingness": missingness, "exclusions": exclusions, "warnings": warnings}


def _select_contrasts(candidates):
    conditions = {item["trial"]["condition"] for item in candidates}
    chosen = [name for name, (a, b) in CONTRASTS.items() if a in conditions and b in conditions]
    # With no computable contrast, report everything as incomplete for the primary one.
    return chosen or [PRIMARY_CONTRAST]


def _contrast_sets(buckets, contrasts, label, exclusions):
    """Per contrast: complete (both ok) pairs with diffs, and attempted (both recorded) pairs."""
    complete = {name: defaultdict(list) for name in contrasts}
    attempted = {name: defaultdict(list) for name in contrasts}
    for bucket in buckets:
        if bucket["rejected"]:
            continue
        missing, not_ok = [], []
        for name in contrasts:
            a_name, b_name = CONTRASTS[name]
            item_a, item_b = bucket["items"].get(a_name), bucket["items"].get(b_name)
            if item_a is None and item_b is None:
                missing.append(name)
                continue
            reference = item_a or item_b
            base = {"pair_id": bucket["pair_id"], "subject": bucket["subject"],
                    "subjects": [bucket["subject"]] if bucket["subject"] is not None else [],
                    "cluster": reference["trial"]["cluster_id"]}
            attempted[name][bucket["group"]].append({**base, "a": item_a, "b": item_b})
            if item_a is None or item_b is None:
                missing.append(name)
                continue
            if item_a["metrics"] is None or item_b["metrics"] is None:
                not_ok.append(name)
                continue
            diff = {metric: item_a["metrics"][metric] - item_b["metrics"][metric] for metric in METRICS}
            complete[name][bucket["group"]].append({**base, "a": item_a["trial"], "b": item_b["trial"], "diff": diff})
        indices = ",".join(str(item["index"]) for item in bucket["records"])
        for reason, names in (("incomplete_pair", missing), ("non_ok_pair", not_ok)):
            if names:
                exclusions.append({"source": label, "record_index": indices, "pair_id": bucket["pair_id"],
                                   "contrast": ";".join(names), "reason": reason, "detail": ""})
    return {"complete": complete, "attempted": attempted}


def _unpaired_sets(buckets, label, exclusions, subjects=None):
    """Between-subject estimator: each pair's effect is mean(event responders) - mean(neutral responders)."""
    arms = defaultdict(lambda: {"event": [], "neutral": []})
    for bucket in buckets:
        if bucket["rejected"] or (subjects is not None and bucket["subject"] not in subjects):
            continue
        present = [name for name in ("event", "neutral") if name in bucket["items"]]
        if len(present) == 2:
            raise ValueError(f"human_design=unpaired but subject {bucket['subject']!r} answered both conditions of pair {bucket['pair_id']!r}")
        for name in present:
            arms[(bucket["group"], bucket["pair_id"])][name].append(bucket["items"][name])
    complete = {PRIMARY_CONTRAST: defaultdict(list)}
    attempted = {PRIMARY_CONTRAST: defaultdict(list)}
    for (group, pair_id), sides in sorted(arms.items(), key=lambda entry: _canonical(entry[0])):
        event, neutral = sides["event"], sides["neutral"]
        indices = ",".join(str(item["index"]) for item in event + neutral)
        reason = None
        if not event or not neutral:
            reason = "incomplete_pair"
        else:
            attempted[PRIMARY_CONTRAST][group].append({"pair_id": pair_id})
            first_event, first_neutral = event[0]["trial"], neutral[0]["trial"]
            if (any(not _same_task(first_event, item["trial"]) for item in event[1:]) or
                    any(not _same_task(first_neutral, item["trial"]) for item in neutral[1:]) or
                    any(first_event.get(field) != first_neutral.get(field) for field in META_FIELDS)):
                reason = "inconsistent_trial_across_subjects"
            elif not any(item["metrics"] for item in event) or not any(item["metrics"] for item in neutral):
                reason = "non_ok_pair"
        if reason:
            exclusions.append({"source": label, "record_index": indices, "pair_id": pair_id,
                               "contrast": PRIMARY_CONTRAST, "reason": reason, "detail": ""})
            continue
        event_ok = [(item["subject"], item["metrics"]) for item in event if item["metrics"]]
        neutral_ok = [(item["subject"], item["metrics"]) for item in neutral if item["metrics"]]
        diff = {metric: fmean([m[metric] for _, m in event_ok]) - fmean([m[metric] for _, m in neutral_ok])
                for metric in METRICS}
        complete[PRIMARY_CONTRAST][group].append({
            "pair_id": pair_id, "subject": None, "cluster": event[0]["trial"]["cluster_id"],
            "subjects": sorted({subject for subject, _ in event_ok + neutral_ok}),
            "a": event[0]["trial"], "b": neutral[0]["trial"], "diff": diff,
            "event_resp": event_ok, "neutral_resp": neutral_ok})
    return {"complete": complete, "attempted": attempted}


def _resolve_human_design(buckets, requested):
    if requested != "auto":
        return requested
    both, single = set(), set()
    for bucket in buckets:
        if bucket["rejected"] or bucket["subject"] is None:
            continue
        present = [name for name in ("event", "neutral") if name in bucket["items"]]
        if len(present) == 2:
            both.add(bucket["subject"])
        elif len(present) == 1:
            single.add(bucket["subject"])
    if single and not both:
        return "unpaired"
    if both and not single:
        return "within"
    if not both and not single:
        return "within"
    raise ValueError("human_design=auto cannot decide: some subjects answered both conditions of a pair and others a single one; choose --human-design within or unpaired explicitly")


def _human_sets(buckets, design, label, exclusions, subjects=None):
    selected = [bucket for bucket in buckets if subjects is None or bucket["subject"] in subjects]
    if design == "unpaired":
        return _unpaired_sets(selected, label, exclusions)
    return _contrast_sets(selected, [PRIMARY_CONTRAST], label, exclusions)


def _coverage(prepared, sets, contrasts, unpaired=False):
    rows = []
    for group in sorted(prepared["seen"], key=_canonical):
        keys = prepared["seen"][group]
        observed = len({pair_id for _subject, pair_id in keys}) if unpaired else len(keys)
        for name in contrasts:
            complete = sets["complete"][name].get(group, [])
            attempted = sets["attempted"][name].get(group, [])
            rows.append({**_group_row(group), "source": prepared["label"], "contrast": name,
                         "observed_pairs": observed, "attempted_pairs": len(attempted),
                         "complete_pairs": len(complete), "excluded_pairs": observed - len(complete),
                         "complete_pair_coverage": len(complete) / observed if observed else None,
                         "n_clusters": len({pair["cluster"] for pair in complete})})
    return rows


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def _effect_row(group, source, contrast, metric, stats, pairs, n_clusters, bootstrap):
    estimate, low, high, p_value = stats
    return {**_group_row(group), "source": source, "metric": metric, "unit": UNITS[metric],
            "contrast": contrast, "estimate": estimate, "ci_low": low, "ci_high": high,
            "n_pairs": len(pairs), "n_clusters": n_clusters,
            "n_subjects": len({subject for pair in pairs for subject in pair["subjects"]}),
            "ci_method": "scenario_cluster_percentile" if low is not None else "unavailable",
            "ci_unavailable_reason": _ci_reason(low, bootstrap),
            **{**_inference_blank(metric), "p_value": p_value}}


def _effects(prepared, sets, contrasts, bootstrap, seed):
    source = prepared["label"]
    rows = []
    for group in sorted(prepared["seen"], key=_canonical):
        for contrast in contrasts:
            pairs = sets["complete"][contrast].get(group, [])
            values = {metric: _cluster_values(pairs, metric) for metric in METRICS}
            stats = _estimate_all(values, bootstrap, _rng(seed, [source, contrast, group]), inferential=source == "model")
            rows.extend(_effect_row(group, source, contrast, metric, stats[metric], pairs,
                                    len(values[metric]), bootstrap) for metric in METRICS)
    return rows


def _interactions(prepared, sets, contrasts, bootstrap, seed, exclusions):
    source = prepared["label"]
    tables = {spec[3]: [] for spec in INTERACTIONS}
    for field, label, reference, table in INTERACTIONS:
        index = GROUP_FIELDS.index(field)
        omit = field if field in TASK_FIELDS else None
        for contrast in contrasts:
            groups = sets["complete"][contrast]
            for group in sorted(prepared["seen"], key=_canonical):
                if group[index] == reference:
                    continue
                base = (*group[:index], reference, *group[index + 1:])
                if base not in prepared["seen"]:
                    continue
                left, right = defaultdict(list), defaultdict(list)
                for pair in groups.get(group, []):
                    left[_task_key(pair, omit)].append(pair)
                for pair in groups.get(base, []):
                    right[_task_key(pair, omit)].append(pair)
                matched = []
                ambiguous = 0
                for key in sorted(left.keys() & right.keys(), key=_canonical):
                    if len(left[key]) != 1 or len(right[key]) != 1:
                        ambiguous += 1
                        exclusions.append({"source": source, "record_index": "", "contrast": contrast,
                                           "pair_id": ",".join(p["pair_id"] for p in left[key] + right[key]),
                                           "reason": f"ambiguous_{label}_task_match", "detail": ""})
                        continue
                    target, baseline = left[key][0], right[key][0]
                    if (target["a"].get("economics") != baseline["a"].get("economics") or
                            target["a"].get("data_sha256") != baseline["a"].get("data_sha256")):
                        exclusions.append({"source": source, "record_index": "", "contrast": contrast,
                                           "pair_id": target["pair_id"] + "," + baseline["pair_id"],
                                           "reason": f"{label}_material_metadata_mismatch", "detail": "economics or material identity differs"})
                        continue
                    matched.append({**target, "diff": {m: target["diff"][m] - baseline["diff"][m] for m in METRICS}})
                values = {metric: _cluster_values(matched, metric) for metric in METRICS}
                stats = _estimate_all(values, bootstrap, _rng(seed, ["interaction", source, label, contrast, group]), inferential=source == "model")
                for metric in METRICS:
                    estimate, low, high, p_value = stats[metric]
                    tables[table].append({
                        **_group_row(group), "source": source, "dimension": label,
                        "reference_level": reference, "level": group[index], "metric": metric,
                        "unit": UNITS[metric], "contrast": contrast,
                        "interaction": f"{label}_effect_minus_reference_{label}_effect",
                        "estimate": estimate, "ci_low": low, "ci_high": high,
                        "matched_pairs": len(matched), "n_clusters": len(values[metric]),
                        "unmatched_tasks": len(left.keys() - right.keys()),
                        "unmatched_reference_tasks": len(right.keys() - left.keys()),
                        "ambiguous_task_matches": ambiguous,
                        "ci_method": "paired_scenario_cluster_percentile" if low is not None else "unavailable",
                        "ci_unavailable_reason": _ci_reason(low, bootstrap),
                        **{**_inference_blank(metric), "p_value": p_value}})
    return tables


def _scenario_tables(prepared, sets, contrasts):
    """Per-scenario contrast estimates plus a cross-scenario robustness summary."""
    source = prepared["label"]
    effect_rows, summary_rows = [], []
    for group in sorted(prepared["seen"], key=_canonical):
        for contrast in contrasts:
            pairs = sets["complete"][contrast].get(group, [])
            if not pairs:
                continue
            by_cluster = defaultdict(list)
            for pair in pairs:
                by_cluster[pair["cluster"]].append(pair)
            for metric in METRICS:
                values = _cluster_values(pairs, metric)
                for cluster in sorted(values):
                    effect_rows.append({**_group_row(group), "source": source, "metric": metric,
                                        "unit": UNITS[metric], "contrast": contrast, "cluster_id": cluster,
                                        "estimate": values[cluster], "n_pairs": len(by_cluster[cluster])})
                pooled = fmean(values.values())
                signs = [(v > 0) - (v < 0) for v in values.values()]
                pooled_sign = (pooled > 0) - (pooled < 0)
                summary_rows.append({**_group_row(group), "source": source, "metric": metric,
                                     "unit": UNITS[metric], "contrast": contrast,
                                     "n_scenarios": len(values), "pooled_estimate": pooled,
                                     "share_same_sign": sum(1 for s in signs if s == pooled_sign) / len(signs),
                                     "min_scenario_estimate": min(values.values()),
                                     "max_scenario_estimate": max(values.values())})
    return effect_rows, summary_rows


def _template_effects(prepared, sets, contrasts, bootstrap, seed):
    source = prepared["label"]
    templates = {item["trial"]["template_id"] for item in prepared["candidates"]}
    if len(templates) < 2:
        return []
    rows = []
    for group in sorted(prepared["seen"], key=_canonical):
        for contrast in contrasts:
            pairs = sets["complete"][contrast].get(group, [])
            for template in sorted(templates):
                subset = [pair for pair in pairs if pair["a"]["template_id"] == template]
                values = {metric: _cluster_values(subset, metric) for metric in METRICS}
                stats = _estimate_all(values, bootstrap, _rng(seed, ["template", source, contrast, group, template]))
                for metric in METRICS:
                    estimate, low, high, _p = stats[metric]
                    rows.append({**_group_row(group), "source": source, "template_id": template,
                                 "metric": metric, "unit": UNITS[metric], "contrast": contrast,
                                 "estimate": estimate, "ci_low": low, "ci_high": high,
                                 "n_pairs": len(subset), "n_clusters": len(values[metric]),
                                 "ci_method": "scenario_cluster_percentile" if low is not None else "unavailable",
                                 "ci_unavailable_reason": _ci_reason(low, bootstrap)})
    return rows


def _return_claim_levels(prepared):
    """Describe valid return claims in each condition, including errors shared by both arms."""
    groups = defaultdict(list)
    for bucket in prepared["buckets"]:
        if bucket["rejected"]:
            continue
        for condition, item in bucket["items"].items():
            if item["metrics"] is not None:
                groups[(bucket["group"], condition)].append({
                    "subject": bucket["subject"], "cluster": item["trial"]["cluster_id"],
                    "diff": item["metrics"],
                })
    rows = []
    for (group, condition), items in sorted(groups.items(), key=lambda entry: _canonical(entry[0])):
        for metric in RETURN_METRICS:
            values = _cluster_values(items, metric)
            rows.append({**_group_row(group), "source": prepared["label"], "condition": condition,
                         "metric": metric, "unit": UNITS[metric], "estimate": fmean(values.values()),
                         "n_records": len(items), "n_clusters": len(values),
                         "n_subjects": len({item["subject"] for item in items if item["subject"] is not None})})
    return rows


def _validity_rows(prepared, sets, contrasts, bootstrap, seed, contrast_effects=True):
    source = prepared["label"]
    rows = []
    tallies = defaultdict(Counter)
    for item in prepared["candidates"]:
        group = item["group"]
        status = item["result"]["status"]
        levels = {"all": "all", "condition": item["trial"]["condition"]}
        for name, index in VALIDITY_DIMENSIONS:
            if isinstance(index, int):
                levels[name] = group[index]
        for dimension, level in levels.items():
            tallies[(group[:4], dimension, level)][status] += 1
    for (identity, dimension, level), counts in sorted(tallies.items(), key=lambda entry: _canonical(entry[0])):
        total = sum(counts.values())
        rows.append({**dict(zip(GROUP_FIELDS[:4], identity)), "source": source, "row_type": "valid_rate",
                     "dimension": dimension, "level": level, "n_records": total,
                     **{f"n_{status}": counts[status] for status in STATUSES},
                     "valid_rate": counts["ok"] / total if total else None})
    if not contrast_effects:
        return rows
    for group in sorted(prepared["seen"], key=_canonical):
        for contrast in contrasts:
            attempted = sets["attempted"][contrast].get(group, [])
            pairs = [{**pair, "diff": {"valid": float(pair["a"] is not None and pair["a"]["metrics"] is not None) -
                                       float(pair["b"] is not None and pair["b"]["metrics"] is not None)}} for pair in attempted]
            values = {"valid": _cluster_values(pairs, "valid")}
            estimate, low, high, _p = _estimate_all(values, bootstrap, _rng(seed, ["validity", source, contrast, group]))["valid"]
            rows.append({**_group_row(group), "source": source, "row_type": "contrast_difference",
                         "contrast": contrast, "estimate": estimate, "ci_low": low, "ci_high": high,
                         "n_pairs": len(pairs), "n_clusters": len(values["valid"]),
                         "ci_method": "scenario_cluster_percentile" if low is not None else "unavailable",
                         "ci_unavailable_reason": _ci_reason(low, bootstrap)})
    return rows


def _side_bounds(item, metric, economics):
    if item is not None and item["metrics"] is not None:
        value = item["metrics"][metric]
        return value, value
    if metric == "timing_days":
        return 0.0, float(economics["max_delay_days"])
    if metric == "reserve_shortfall_pct":
        return 0.0, max(100.0, 100.0 * economics["liquidity_need_inr"] / economics["capital_inr"])
    return BOUNDED[metric]


def _bounds_rows(prepared, sets, contrasts):
    """Worst-case (Manski-type) bounds on cluster-averaged contrasts over all attempted pairs."""
    source = prepared["label"]
    bounded = ("gold_pct", "equity_pct", "cash_pct", "buy_now", "timing_days",
               "reserve_shortfall_pct", "reserve_violation", "return_claim_violation")
    rows = []
    for group in sorted(prepared["seen"], key=_canonical):
        for contrast in contrasts:
            attempted = sets["attempted"][contrast].get(group, [])
            complete = sets["complete"][contrast].get(group, [])
            lows, highs = [], []
            for pair in attempted:
                economics = (pair["a"] or pair["b"])["trial"]["economics"]
                low, high = {}, {}
                for metric in bounded:
                    a_low, a_high = _side_bounds(pair["a"], metric, economics)
                    b_low, b_high = _side_bounds(pair["b"], metric, economics)
                    low[metric], high[metric] = a_low - b_high, a_high - b_low
                lows.append({**pair, "diff_low": low})
                highs.append({**pair, "diff_high": high})
            share = (len(attempted) - len(complete)) / len(attempted) if attempted else None
            for metric in METRICS:
                row = {**_group_row(group), "source": source, "metric": metric, "unit": UNITS[metric],
                       "contrast": contrast, "n_pairs_attempted": len(attempted),
                       "n_pairs_complete": len(complete), "share_missing_one_side": share,
                       "n_absent_records": sum(pair["a"] is None or pair["b"] is None for pair in attempted),
                       "n_recorded_nonvalid": sum(item is not None and item["metrics"] is None
                                                 for pair in attempted for item in (pair["a"], pair["b"])),
                       "estimate_complete_case": (fmean(_cluster_values(complete, metric).values())
                                                  if complete else None),
                       "bound_low": None, "bound_high": None, "bounds_exclude_zero": None,
                       "n_clusters_attempted": 0,
                       "skipped_reason": "" if metric in bounded else "unbounded_metric"}
                if metric in bounded and attempted:
                    low_values = _cluster_values(lows, metric, "diff_low")
                    high_values = _cluster_values(highs, metric, "diff_high")
                    row["bound_low"] = fmean(low_values.values())
                    row["bound_high"] = fmean(high_values.values())
                    row["bounds_exclude_zero"] = row["bound_low"] > 0 or row["bound_high"] < 0
                    row["n_clusters_attempted"] = len(low_values)
                rows.append(row)
    return rows


def _same_task(left, right):
    return all(left.get(field) == right.get(field) for field in
               (*PAIR_FIELDS, *OPTIONAL_TASK_FIELDS, "pair_id", "trial_id", "condition", "prompt_sha256", "prompt", "economics", "data_sha256"))


def _human_assignment_diagnostics(study, prepared, roster, selected_splits):
    """Bound assigned-arm means, including participants who never submitted."""
    by_id = {t["trial_id"]: t for t in roster}
    responses = {(i["subject"], i["trial"]["trial_id"]): i for i in prepared["candidates"]}
    arms = defaultdict(lambda: {"event": [], "neutral": []})
    identity = ("human", "human", "self_report", "{}")
    for key, assigned in study["index"].items():
        trial = by_id[assigned["trial_id"]]
        if trial["split"] not in selected_splits or trial["condition"] not in ("event", "neutral"):
            continue
        group = (*identity, *(trial[f] for f in GROUP_FIELDS[4:]))
        arms[group, trial["pair_id"]][trial["condition"]].append((responses.get(key), trial))
    bounded = tuple(BOUNDED) + ("timing_days", "reserve_shortfall_pct")
    pairs = defaultdict(list)
    unassigned = defaultdict(int)
    for (group, pair_id), sides in arms.items():
        if not all(sides.values()):
            unassigned[group] += 1
            continue
        reference = sides["event"][0][1]
        pair = {"cluster": reference["cluster_id"], "subject": None, "pair_id": pair_id,
                "a": reference, "diff_low": {}, "diff_high": {}, "diff": {},
                "n_absent": 0, "n_nonvalid": 0, "n_assigned": 0}
        for values in sides.values():
            pair["n_assigned"] += len(values)
            pair["n_absent"] += sum(item is None for item, _ in values)
            pair["n_nonvalid"] += sum(item is not None and item["metrics"] is None for item, _ in values)
        for metric in bounded:
            limits = {name: [_side_bounds(item, metric, trial["economics"]) for item, trial in values]
                      for name, values in sides.items()}
            pair["diff_low"][metric] = fmean(v[0] for v in limits["event"]) - fmean(v[1] for v in limits["neutral"])
            pair["diff_high"][metric] = fmean(v[1] for v in limits["event"]) - fmean(v[0] for v in limits["neutral"])
        pair["diff"]["valid"] = (fmean(float(i is not None and i["metrics"] is not None) for i, _ in sides["event"])
                                     - fmean(float(i is not None and i["metrics"] is not None) for i, _ in sides["neutral"]))
        pairs[group].append(pair)
    bounds, validity = [], []
    for group in sorted(set(pairs) | set(unassigned), key=_canonical):
        tasks = pairs[group]
        for metric in bounded:
            low = fmean(_cluster_values(tasks, metric, "diff_low").values()) if tasks else None
            high = fmean(_cluster_values(tasks, metric, "diff_high").values()) if tasks else None
            bounds.append({**_group_row(group), "source": "human_assignment", "metric": metric,
                           "unit": UNITS[metric], "contrast": PRIMARY_CONTRAST,
                           "bound_low": low, "bound_high": high,
                           "bounds_exclude_zero": low > 0 or high < 0 if tasks else None,
                           "n_pairs_attempted": len(tasks), "n_pairs_complete": sum(not p["n_absent"] and not p["n_nonvalid"] for p in tasks),
                           "n_absent_records": sum(p["n_absent"] for p in tasks),
                           "n_recorded_nonvalid": sum(p["n_nonvalid"] for p in tasks),
                           "n_clusters_attempted": len({p["cluster"] for p in tasks}),
                           "n_tasks_missing_assigned_arm": unassigned[group],
                           "estimand": "equal_planned_tasks_assigned_subject_arm_means",
                           "skipped_reason": "no_tasks_assigned_both_arms" if not tasks else ""})
        validity.append({**_group_row(group), "source": "human_assignment", "row_type": "assigned_validity_difference",
                         "contrast": PRIMARY_CONTRAST,
                         "estimate": fmean(_cluster_values(tasks, "valid").values()) if tasks else None,
                         "n_pairs": len(tasks), "n_clusters": len({p["cluster"] for p in tasks}),
                         "ci_low": None, "ci_high": None, "ci_unavailable_reason": "descriptive_assigned_sample_only"})
    return bounds, validity


def _comparison_row(group, design, estimate, low, high, draws, bootstrap, matched_model, matched_units,
                    clusters, subjects, metric, unmatched, unavailable_reason=None):
    return {**_group_row(group), "metric": metric, "unit": UNITS[metric],
            "contrast": "model_event_effect_minus_human_event_effect", "human_design": design,
            "estimate": estimate, "ci_low": low, "ci_high": high,
            "matched_model_pairs": matched_model, "matched_human_units": matched_units,
            "n_clusters": clusters, "n_subjects": subjects, "bootstrap_valid_draws": draws,
            "ci_method": ("crossed_subject_scenario_percentile" if design == "within" else
                          "stratified_subject_scenario_percentile") if low is not None else "unavailable",
            "ci_unavailable_reason": "" if low is not None else (unavailable_reason or
                ("bootstrap_disabled" if not bootstrap else "requires_two_subjects_and_two_clusters")),
            "unmatched_model_pairs": unmatched}


def _human_comparisons(models, humans, design, bootstrap, seed, exclusions):
    """model_event_effect - human_event_effect on identical trials (event_minus_neutral only)."""
    human_index = defaultdict(list)
    for pairs in humans.values():
        for pair in pairs:
            human_index[(pair["a"]["trial_id"], pair["b"]["trial_id"])].append(pair)
    rows = []
    for group in sorted(models, key=_canonical):
        matches = []
        matched_model_pairs = set()
        for model in models[group]:
            ids = (model["a"]["trial_id"], model["b"]["trial_id"])
            for human in human_index.get(ids, []):
                if not (_same_task(model["a"], human["a"]) and _same_task(model["b"], human["b"])):
                    exclusions.append({"source": "comparison", "record_index": "", "contrast": PRIMARY_CONTRAST,
                                       "pair_id": model["pair_id"], "reason": "human_task_metadata_mismatch", "detail": ""})
                    continue
                matched_model_pairs.add(ids)
                matches.append((model, human))
        unmatched = len(models[group]) - len(matched_model_pairs)
        if design == "unpaired":
            rows.extend(_comparison_unpaired(group, matches, matched_model_pairs, unmatched, bootstrap, seed))
        else:
            rows.extend(_comparison_within(group, matches, matched_model_pairs, unmatched, bootstrap, seed))
    return rows


def _comparison_within(group, matches, matched_model_pairs, unmatched, bootstrap, seed):
    pairs = [{**human, "diff": {m: model["diff"][m] - human["diff"][m] for m in METRICS}}
             for model, human in matches]
    rows = []
    for metric in METRICS:
        cells = defaultdict(list)
        for pair in pairs:
            cells[(pair["cluster"], pair["subject"])].append(pair["diff"][metric])
        cells = {key: fmean(values) for key, values in cells.items()}
        clusters = sorted({key[0] for key in cells})
        subjects = sorted({key[1] for key in cells})
        values = _cluster_values(pairs, metric)
        estimate = fmean(values.values()) if values else None
        draws = []
        if len(clusters) >= 2 and len(subjects) >= 2 and bootstrap:
            rng = _rng(seed, ["human", group, metric])
            for _ in range(bootstrap):
                cluster_counts = Counter(rng.choices(clusters, k=len(clusters)))
                subject_counts = Counter(rng.choices(subjects, k=len(subjects)))
                sampled = []
                for cluster, cluster_weight in cluster_counts.items():
                    observed = [(value, subject_counts[subject]) for (c, subject), value in cells.items()
                                if c == cluster and subject_counts[subject]]
                    denominator = sum(weight for _, weight in observed)
                    if denominator:
                        mean = sum(value * weight for value, weight in observed) / denominator
                        sampled.extend([mean] * cluster_weight)
                if sampled:
                    draws.append(fmean(sampled))
        low, high = _interval(draws)
        rows.append(_comparison_row(group, "within", estimate, low, high, len(draws), bootstrap,
                                    len(matched_model_pairs), len(pairs), len(clusters), len(subjects),
                                    metric, unmatched))
    return rows


def _comparison_unpaired(group, matches, matched_model_pairs, unmatched, bootstrap, seed):
    pairs = [{"cluster": human["cluster"], "subject": None, "model": model["diff"],
              "event": human["event_resp"], "neutral": human["neutral_resp"],
              "diff": {m: model["diff"][m] - human["diff"][m] for m in METRICS}}
             for model, human in matches]
    by_cluster = defaultdict(list)
    for pair in pairs:
        by_cluster[pair["cluster"]].append(pair)
    clusters = sorted(by_cluster)
    arms = defaultdict(set)
    for pair in pairs:
        for arm in ("event", "neutral"):
            for subject, _metrics_ in pair[arm]:
                arms[subject].add(arm)
    # Subjects are resampled within strata defined by which arm(s) they appeared in,
    # so a between-subject design keeps arm sizes fixed and a counterbalanced subject
    # who answered different clusters in both arms stays one dependent unit.
    strata = defaultdict(list)
    for subject in sorted(arms):
        strata[tuple(sorted(arms[subject]))].append(subject)
    estimates = {metric: _cluster_values(pairs, metric) for metric in METRICS}
    draws = {metric: [] for metric in METRICS}
    valid = 0
    if not bootstrap:
        unavailable_reason = "bootstrap_disabled"
    elif len(clusters) < 2 or len(arms) < 2:
        unavailable_reason = "requires_two_subjects_and_two_clusters"
    elif any(len(subjects) < 2 for subjects in strata.values()):
        unavailable_reason = "requires_two_subjects_per_arm_stratum"
    else:
        unavailable_reason = None
    if unavailable_reason is None:
        rng = _rng(seed, ["human-unpaired", group])
        for _ in range(bootstrap):
            cluster_counts = Counter(rng.choices(clusters, k=len(clusters)))
            weights = Counter()
            for key in sorted(strata):
                weights.update(rng.choices(strata[key], k=len(strata[key])))
            sampled = {metric: [] for metric in METRICS}
            for cluster in sorted(cluster_counts):
                surviving = {metric: [] for metric in METRICS}
                for pair in by_cluster[cluster]:
                    event_weight = sum(weights[s] for s, _ in pair["event"])
                    neutral_weight = sum(weights[s] for s, _ in pair["neutral"])
                    if not event_weight or not neutral_weight:
                        continue
                    for metric in METRICS:
                        human = (sum(weights[s] * m[metric] for s, m in pair["event"]) / event_weight -
                                 sum(weights[s] * m[metric] for s, m in pair["neutral"]) / neutral_weight)
                        surviving[metric].append(pair["model"][metric] - human)
                if surviving[METRICS[0]]:
                    for metric in METRICS:
                        sampled[metric].extend([fmean(surviving[metric])] * cluster_counts[cluster])
            if sampled[METRICS[0]]:
                valid += 1
                for metric in METRICS:
                    draws[metric].append(fmean(sampled[metric]))
    rows = []
    for metric in METRICS:
        low, high = _interval(draws[metric])
        estimate = fmean(estimates[metric].values()) if estimates[metric] else None
        rows.append(_comparison_row(group, "unpaired", estimate, low, high, valid, bootstrap,
                                    len(matched_model_pairs), len(pairs), len(clusters), len(arms),
                                    metric, unmatched, unavailable_reason or "no_valid_bootstrap_draws"))
    return rows


# ---------------------------------------------------------------------------
# Preregistration and multiplicity
# ---------------------------------------------------------------------------

def _load_prereg(path):
    raw = Path(path).read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    try:
        data = strict_json(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError(f"{path}: invalid preregistration JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: preregistration must be a JSON object")
    unknown = set(data) - PREREG_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown preregistration keys: {sorted(unknown)}")
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    if "_note" in data and not isinstance(data["_note"], str):
        raise ValueError(f"{path}: _note must be text")
    tolerance = _return_tolerance(data["return_tolerance_pp"]) if "return_tolerance_pp" in data else None
    primary = data.get("primary")
    if not isinstance(primary, list) or not primary:
        raise ValueError(f"{path}: primary must be a nonempty list")
    if data.get("primary_adjustment", "holm") != "holm":
        raise ValueError(f"{path}: primary_adjustment must be 'holm'")
    exploratory = data.get("exploratory_adjustment", "benjamini_hochberg")
    if exploratory not in EXPLORATORY_ADJUSTMENTS:
        raise ValueError(f"{path}: exploratory_adjustment must be one of {EXPLORATORY_ADJUSTMENTS}")
    entries, ids = [], set()
    for number, entry in enumerate(primary, 1):
        where = f"{path}: primary[{number}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} must be an object")
        extra = set(entry) - PREREG_ENTRY_KEYS
        if extra:
            raise ValueError(f"{where}: unknown keys {sorted(extra)}")
        absent = PREREG_ENTRY_KEYS - {"filters", "expected_tests"} - set(entry)
        if absent:
            raise ValueError(f"{where}: missing keys {sorted(absent)}")
        if not isinstance(entry["id"], str) or not entry["id"] or entry["id"] in ids:
            raise ValueError(f"{where}: id must be a unique nonempty string")
        ids.add(entry["id"])
        if entry["table"] not in FAMILY_TABLES:
            raise ValueError(f"{where}: table must be one of {FAMILY_TABLES}")
        if entry["metric"] not in METRICS:
            raise ValueError(f"{where}: unknown metric {entry['metric']!r}")
        if entry["metric"] in DERIVED_FROM_ALLOCATION:
            raise ValueError(f"{where}: {entry['metric']} is derived from the allocation and cannot be primary")
        if entry["metric"] in REDUNDANT_CONTRAST_METRICS:
            raise ValueError(f"{where}: {entry['metric']} duplicates expected_return_pct in a contrast; "
                             "use return_claim_absolute_error_pp or return_claim_violation")
        if entry["contrast"] not in CONTRASTS:
            raise ValueError(f"{where}: contrast must be one of {tuple(CONTRASTS)}")
        mme = entry["minimum_meaningful_effect"]
        if isinstance(mme, bool) or not isinstance(mme, (int, float)) or not math.isfinite(mme) or mme <= 0:
            raise ValueError(f"{where}: minimum_meaningful_effect must be a positive finite number")
        expected = entry.get("expected_tests")
        if "expected_tests" in entry and (isinstance(expected, bool) or not isinstance(expected, int) or expected < 1):
            raise ValueError(f"{where}: expected_tests must be a positive integer")
        filters = entry.get("filters", {})
        if not isinstance(filters, dict):
            raise ValueError(f"{where}: filters must be an object")
        for key, value in filters.items():
            if key not in PREREG_FILTER_KEYS:
                raise ValueError(f"{where}: unknown filter {key!r}; allowed: {PREREG_FILTER_KEYS}")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{where}: filter {key!r} must be a nonempty string")
        entries.append({"id": entry["id"], "table": entry["table"], "metric": entry["metric"],
                        "contrast": entry["contrast"], "filters": dict(filters),
                        "minimum_meaningful_effect": float(mme), "expected_tests": expected})
    return {"sha256": sha256, "path": str(path), "primary": entries,
            "primary_adjustment": "holm", "exploratory_adjustment": exploratory,
            "return_tolerance_pp": tolerance,
            **{key: data.get(key) for key in ("inference_method", "inference_assumption", "registration",
                                             "evaluation_splits", "target_population", "template_weighting")}}


def _holm(p_values):
    order = sorted(range(len(p_values)), key=lambda i: (p_values[i], i))
    adjusted, running = [None] * len(p_values), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _benjamini_hochberg(p_values):
    m = len(p_values)
    order = sorted(range(m), key=lambda i: (p_values[i], i), reverse=True)
    adjusted, running = [None] * m, 1.0
    for k, i in enumerate(order):
        running = min(running, p_values[i] * m / (m - k))
        adjusted[i] = min(1.0, running)
    return adjusted


def _adjust_family(rows, method, family, missing_tests=0):
    """Keep unavailable planned tests in the family without reporting imputed p-values."""
    adjusters = {"holm": _holm, "benjamini_hochberg": _benjamini_hochberg,
                 "none": lambda values: list(values)}
    p_values = [row["p_value"] if row["p_value"] is not None else 1.0 for row in rows]
    adjusted = adjusters[method](p_values + [1.0] * missing_tests)
    for row, value in zip(rows, adjusted):
        row["p_adjusted"] = value if row["p_value"] is not None else None
    for row in rows:
        row["family"] = family
        row["adjustment"] = method
        row["n_family"] = len(rows) + missing_tests
    return len(rows) + missing_tests


def _apply_multiplicity(tables, prereg):
    entries = prereg["primary"] if prereg else []
    matched = {entry["id"]: [] for entry in entries}
    primary, exploratory = [], []
    for name in FAMILY_TABLES:
        for row in tables[name]:
            if row["metric"] in DERIVED_FROM_ALLOCATION:
                row.update(family="derived_from_allocation", adjustment="none")
                continue
            if row["metric"] in REDUNDANT_CONTRAST_METRICS:
                row.update(family="redundant_return_contrast", adjustment="none")
                continue
            hits = [entry for entry in entries if entry["table"] == name and
                    entry["metric"] == row["metric"] and entry["contrast"] == row["contrast"] and
                    all(str(row.get(key)) == value for key, value in entry["filters"].items())]
            if len(hits) > 1:
                raise ValueError(f"Row {name}/{row['metric']}/{row['contrast']} matches several primary entries: "
                                 f"{[hit['id'] for hit in hits]}; make primary filters disjoint")
            if hits:
                entry = hits[0]
                row["primary_id"] = entry["id"]
                row["minimum_meaningful_effect"] = entry["minimum_meaningful_effect"]
                low, high, mme = row["ci_low"], row["ci_high"], entry["minimum_meaningful_effect"]
                if low is not None:
                    row["exceeds_mme"] = low > mme or high < -mme
                    row["within_mme"] = low >= -mme and high <= mme
                matched[entry["id"]].append(row)
                primary.append(row)
            else:
                exploratory.append(row)
    planned = {}
    for entry in entries:
        count = len(matched[entry["id"]])
        expected = entry.get("expected_tests")
        if expected is not None and count > expected:
            raise ValueError(f"Primary entry {entry['id']!r} matched {count} rows, exceeding "
                             f"its expected_tests={expected}; check the registered filters")
        planned[entry["id"]] = expected if expected is not None else max(1, count)
    missing_primary = sum(planned[entry["id"]] - len(matched[entry["id"]]) for entry in entries)
    n_primary = _adjust_family(primary, "holm", "primary", missing_primary)
    method = prereg["exploratory_adjustment"] if prereg else "benjamini_hochberg"
    n_exploratory = _adjust_family(exploratory, method, "exploratory")
    summary = {
        "provided": prereg is not None,
        "sha256": prereg["sha256"] if prereg else None,
        "path": prereg["path"] if prereg else None,
        "primary_adjustment": "holm", "exploratory_adjustment": method,
        "entries": [{**entry, "n_rows_matched": len(matched[entry["id"]]),
                     "n_planned_tests": planned[entry["id"]],
                     "n_tests_with_p_value": sum(1 for row in matched[entry["id"]] if row["p_value"] is not None)}
                    for entry in entries],
        "unmatched_primary": [entry["id"] for entry in entries if not matched[entry["id"]]],
        "n_primary_tests": n_primary, "n_exploratory_tests": n_exploratory,
        "n_primary_tests_with_p_value": sum(row["p_value"] is not None for row in primary),
        "n_unavailable_primary_tests": n_primary - sum(row["p_value"] is not None for row in primary),
        "n_redundant_return_contrasts_excluded": sum(
            row["metric"] in REDUNDANT_CONTRAST_METRICS for name in FAMILY_TABLES for row in tables[name]),
        "n_derived_from_allocation_excluded": sum(
            1 for name in FAMILY_TABLES for row in tables[name] if row["metric"] in DERIVED_FROM_ALLOCATION),
    }
    return summary


# ---------------------------------------------------------------------------
# Participant covariates and human heterogeneity
# ---------------------------------------------------------------------------

def _check_participant(entry, where):
    if not isinstance(entry, dict) or set(entry) != {*PARTICIPANT_REQUIRED, *PARTICIPANT_OPTIONAL}:
        raise ValueError(f"{where}: participant must have exactly the fields {[*PARTICIPANT_REQUIRED, *PARTICIPANT_OPTIONAL]}")
    if not isinstance(entry["subject_id"], str) or not entry["subject_id"].strip() or entry["subject_id"] != entry["subject_id"].strip():
        raise ValueError(f"{where}: subject_id must be a nonempty trimmed string")
    if entry["survey_language"] not in PARTICIPANT_LANGUAGES:
        raise ValueError(f"{where}: survey_language must be one of {PARTICIPANT_LANGUAGES}")
    for field in ("language_proficiency", "observance_self_report"):
        value = entry[field]
        if type(value) is not int or not 1 <= value <= 5:
            raise ValueError(f"{where}: {field} must be an integer 1-5")
    literacy = entry["financial_literacy"]
    if isinstance(literacy, bool) or not isinstance(literacy, (int, float)) or not math.isfinite(literacy):
        raise ValueError(f"{where}: financial_literacy must be a finite number")
    for field in PARTICIPANT_OPTIONAL:
        if entry[field] is not None and (not isinstance(entry[field], str) or not entry[field].strip()):
            raise ValueError(f"{where}: {field} must be null or nonempty text")


def _participant_from_row(row, where):
    entry = {field: row[field].strip() for field in PARTICIPANT_REQUIRED}
    for field in ("language_proficiency", "observance_self_report"):
        if not re.fullmatch(r"[0-9]+", entry[field]):
            raise ValueError(f"{where}: {field} must be an integer 1-5, got {entry[field]!r}")
        entry[field] = int(entry[field])
    try:
        entry["financial_literacy"] = float(entry["financial_literacy"])
    except ValueError:
        raise ValueError(f"{where}: financial_literacy must be a number, got {entry['financial_literacy']!r}") from None
    for field in PARTICIPANT_OPTIONAL:
        text = (row.get(field) or "").strip()
        entry[field] = text or None
    _check_participant(entry, where)
    return entry


def import_participants(csv_path, output_path, *, assignment_path=None, instrument_path=None):
    """Strictly import self-reported participant covariates; nothing is imputed.

    Required columns: subject_id, survey_language (en|hi|ta|gu|bn),
    language_proficiency (integer 1-5, self-rated in the survey language),
    financial_literacy (numeric score from the researcher's own instrument; its
    scale and validity are not checked here), observance_self_report (integer 1-5,
    personal observance of auspicious calendar periods). Optional: age_band, region.
    Unknown columns, blank required cells and duplicate subject_ids are errors.
    A header-only file is valid.
    """
    participants = []
    seen = set()
    for line, row in _csv_read(csv_path, PARTICIPANT_REQUIRED,
                               allowed=(*PARTICIPANT_REQUIRED, *PARTICIPANT_OPTIONAL)):
        entry = _participant_from_row(row, f"{csv_path}:{line}")
        if entry["subject_id"] in seen:
            raise ValueError(f"{csv_path}:{line}: duplicate subject_id {entry['subject_id']!r}")
        seen.add(entry["subject_id"])
        participants.append(entry)
    by_language = Counter(entry["survey_language"] for entry in participants)
    provenance = {"csv_sha256": hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
                  "assignment_status": "unverified_legacy", "instrument_status": "unverified"}
    if assignment_path:
        from .protocol import load_assignments
        study = load_assignments(assignment_path)
        expected = {row["subject_id"]: row["language"] for row in study["rows"]}
        if any(expected.get(p["subject_id"]) != p["survey_language"] for p in participants):
            raise ValueError("Participant subject/language does not match this study's assignments")
        provenance.update(study_id=study["study_id"], assignment_manifest_sha256=study["manifest_sha256"],
                          assignment_status="bound_to_study")
    if instrument_path:
        instrument_raw = Path(instrument_path).read_bytes()
        instrument = strict_json(instrument_raw.decode("utf-8"))
        if not isinstance(instrument, dict) or any(not str(instrument.get(k, "")).strip()
                                                 for k in ("id", "version", "source", "scoring")):
            raise ValueError("Instrument needs id, version, source and scoring documentation")
        provenance.update(instrument=instrument, instrument_sha256=hashlib.sha256(instrument_raw).hexdigest(),
                          instrument_status="documented_not_independently_validated")
    document = {"schema_version": 1, "n_participants": len(participants), "participants": participants,
                "provenance": provenance,
                "notes": "Self-reported covariates. financial_literacy is a researcher-defined numeric score; instrument and scale are not validated here. Used for descriptive heterogeneity only, never to adjust the model-human comparison."}
    _json_write(output_path, document)
    return {"participants": len(participants), "by_survey_language": dict(sorted(by_language.items())),
            "output": str(output_path), "warnings": [] if assignment_path and instrument_path else
            ["Study and/or instrument provenance unverified; descriptive covariates only."]}


def _load_participants(path):
    try:
        document = strict_json(Path(path).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path}: invalid participants JSON: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1 or \
            not isinstance(document.get("participants"), list):
        raise ValueError(f"{path}: expected an import_participants output (schema_version 1)")
    seen = set()
    for number, entry in enumerate(document["participants"], 1):
        _check_participant(entry, f"{path}: participants[{number}]")
        if entry["subject_id"] in seen:
            raise ValueError(f"{path}: duplicate subject_id {entry['subject_id']!r}")
        seen.add(entry["subject_id"])
    return document["participants"]


def _describe(values):
    return (fmean(values), min(values), max(values)) if values else (None, None, None)


def _participant_tables(participants, observed_subjects):
    by_language = defaultdict(list)
    for entry in participants:
        by_language["all"].append(entry)
        by_language[entry["survey_language"]].append(entry)
    rows = []
    for language in ("all", *PARTICIPANT_LANGUAGES):
        group = by_language.get(language, [])
        if not group and language != "all":
            continue
        proficiency = _describe([e["language_proficiency"] for e in group])
        literacy_values = [e["financial_literacy"] for e in group]
        literacy = _describe(literacy_values)
        observance = _describe([e["observance_self_report"] for e in group])
        rows.append({
            "survey_language": language, "n_participants": len(group),
            "n_with_responses": sum(1 for e in group if e["subject_id"] in observed_subjects),
            "language_proficiency_mean": proficiency[0], "language_proficiency_min": proficiency[1],
            "language_proficiency_max": proficiency[2], "financial_literacy_mean": literacy[0],
            "financial_literacy_median": median(literacy_values) if literacy_values else None,
            "financial_literacy_min": literacy[1], "financial_literacy_max": literacy[2],
            "observance_mean": observance[0],
            "observance_high_n": sum(1 for e in group if e["observance_self_report"] >= 4),
            "observance_middle_n": sum(1 for e in group if e["observance_self_report"] == 3),
            "observance_low_n": sum(1 for e in group if e["observance_self_report"] <= 2)})
    return rows


def _heterogeneity(buckets, design, participants, bootstrap, seed, exclusions):
    by_id = {entry["subject_id"]: entry for entry in participants}
    observed = sorted({bucket["subject"] for bucket in buckets if bucket["subject"] is not None})
    for subject in observed:
        if subject not in by_id:
            exclusions.append({"source": "participants", "record_index": "", "pair_id": "", "contrast": "",
                               "reason": "subject_missing_from_participants_file", "detail": subject})
    known = [subject for subject in observed if subject in by_id]
    high = {s for s in known if by_id[s]["observance_self_report"] >= 4}
    low = {s for s in known if by_id[s]["observance_self_report"] <= 2}
    middle = len(known) - len(high) - len(low)
    literacy = [by_id[s]["financial_literacy"] for s in known]
    cut = median(literacy) if literacy else None
    splits = [
        ("observance_self_report", "high", "observance_self_report >= 4", None, high, middle),
        ("observance_self_report", "low", "observance_self_report <= 2", None, low, middle),
        ("financial_literacy", "high", "financial_literacy > sample median", cut,
         {s for s in known if by_id[s]["financial_literacy"] > cut} if known else set(), 0),
        ("financial_literacy", "low", "financial_literacy <= sample median", cut,
         {s for s in known if by_id[s]["financial_literacy"] <= cut} if known else set(), 0),
    ]
    rows = []
    for covariate, level, rule, split_value, subjects, n_middle in splits:
        for language in ("all", *PARTICIPANT_LANGUAGES):
            members = {s for s in subjects if language == "all" or by_id[s]["survey_language"] == language}
            if not members and language != "all":
                continue
            sets = _human_sets(buckets, design, "human_heterogeneity", [], subjects=members)
            pairs = [pair for group in sets["complete"][PRIMARY_CONTRAST].values() for pair in group]
            values = {metric: _cluster_values(pairs, metric) for metric in METRICS}
            stats = _estimate_all(values, bootstrap, _rng(seed, ["heterogeneity", covariate, level, language]), inferential=False)
            for metric in METRICS:
                estimate, low_ci, high_ci, _p = stats[metric]
                rows.append({"covariate": covariate, "level": level, "survey_language": language,
                             "split_rule": rule, "split_value": split_value, "metric": metric,
                             "unit": UNITS[metric], "contrast": PRIMARY_CONTRAST, "human_design": design,
                             "estimate": estimate, "ci_low": low_ci, "ci_high": high_ci,
                             "n_subjects_in_split": len(members),
                             "n_subjects_contributing": len({s for pair in pairs for s in pair["subjects"]}),
                             "n_pairs": len(pairs), "n_clusters": len(values[metric]),
                             "n_middle_excluded": n_middle,
                             "ci_method": "scenario_cluster_percentile" if low_ci is not None else "unavailable",
                             "ci_unavailable_reason": _ci_reason(low_ci, bootstrap)})
    return rows, set(observed)


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def _design_summary(prepared):
    trials = {}
    for item in prepared["candidates"]:
        trials.setdefault(item["trial"]["trial_id"], item["trial"])
    scenarios = sorted({trial["scenario_id"] for trial in trials.values()})
    prompts = {trial.get("prompt_sha256") or hashlib.sha256(str(trial.get("prompt")).encode()).hexdigest()
               for trial in trials.values()}
    summary = {
        "n_trials_observed": len(trials), "n_scenarios": len(scenarios), "scenarios": scenarios,
        "templates": sorted({trial["template_id"] for trial in trials.values()}),
        "replicates": sorted({trial["replicate"] for trial in trials.values()}),
        "conditions": sorted({trial["condition"] for trial in trials.values()}),
        "distinct_prompt_texts": len(prompts), "warnings": [],
    }
    if len(scenarios) < MIN_SCENARIOS_FOR_GENERAL_CLAIMS:
        summary["warnings"].append(
            f"{prepared['label']}: only {len(scenarios)} distinct scenario(s) (< {MIN_SCENARIOS_FOR_GENERAL_CLAIMS}); "
            "intervals describe only the sampled scenarios, not a wider population of financial situations.")
    return summary


def analyze(records_path, output_dir, *, bootstrap=2000, seed=42, human_path=None, prereg_path=None,
            human_design="auto", participants_path=None, return_tolerance_pp=None, trials_path=None,
            evaluation_splits=("test",), inference_method=None, assignment_path=None):
    plan = _load_prereg(prereg_path) if prereg_path is not None else None
    method = inference_method or (plan.get("inference_method") if plan else None) or "descriptive"
    if method != "descriptive" and (plan is None or plan.get("inference_assumption") != ASSUMPTION):
        raise ValueError("Sign-flip inference requires the registered independent sign-symmetry assumption")
    with inference_scope(method):
        return _analyze(records_path, output_dir, bootstrap=bootstrap, seed=seed, human_path=human_path,
                        prereg_path=prereg_path, human_design=human_design, participants_path=participants_path,
                        return_tolerance_pp=return_tolerance_pp, trials_path=trials_path,
                        evaluation_splits=evaluation_splits, inference_method=method, assignment_path=assignment_path)


def _analyze(records_path, output_dir, *, bootstrap=2000, seed=42, human_path=None, prereg_path=None,
             human_design="auto", participants_path=None, return_tolerance_pp=None, trials_path=None,
             evaluation_splits=("test",), inference_method="descriptive", assignment_path=None):
    """Write descriptive effects and attrition; gate opt-in sign-symmetry inference."""
    if isinstance(bootstrap, bool) or not isinstance(bootstrap, int) or bootstrap < 0:
        raise ValueError("bootstrap must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if human_design not in HUMAN_DESIGNS:
        raise ValueError(f"human_design must be one of {HUMAN_DESIGNS}")
    if participants_path is not None and human_path is None:
        raise ValueError("participants_path requires human_path")
    prereg = _load_prereg(prereg_path) if prereg_path is not None else None
    selected_splits = set(evaluation_splits)
    if not selected_splits or not selected_splits <= {"train", "dev", "test"}:
        raise ValueError("evaluation_splits must select train, dev and/or test")
    if prereg and prereg.get("evaluation_splits") is not None and selected_splits != set(prereg["evaluation_splits"]):
        raise ValueError("Selected splits differ from the prospectively frozen evaluation splits")
    registered_tolerance = prereg.get("return_tolerance_pp") if prereg else None
    if return_tolerance_pp is not None:
        return_tolerance_pp = _return_tolerance(return_tolerance_pp)
        if registered_tolerance is not None and return_tolerance_pp != registered_tolerance:
            raise ValueError("return_tolerance_pp differs from the preregistered tolerance")
    else:
        return_tolerance_pp = (registered_tolerance if registered_tolerance is not None
                               else DEFAULT_RETURN_TOLERANCE_PP)
    participants = _load_participants(participants_path) if participants_path is not None else None
    records = load_records(records_path)
    roster = read_jsonl(trials_path) if trials_path is not None else None
    if roster is not None:
        from .protocol import validate_roster
        validate_trials(roster)
        validate_roster(records, roster)
    records = [r for r in records if not isinstance(r.get("trial"), dict)
               or r["trial"].get("split") not in ("train", "dev", "test")
               or r["trial"].get("split") in selected_splits]
    if inference_method != "descriptive":
        from .protocol import collection_protocol, verify_bound_records
        from .core import load_review_context
        if roster is None:
            raise ValueError("Inference requires the original --trials roster")
        collection_protocol(prereg_path, confirmatory=True)
        validate_trials(roster, confirmatory=True, review_context=load_review_context(trials_path, True))
        verify_bound_records(records, prereg["sha256"], trials=roster,
                             registered_at=prereg["registration"]["registered_at"])
        expected = {t["trial_id"] for t in roster if t["split"] in selected_splits}
        observed = {r["trial"]["trial_id"] for r in records}
        if observed != expected or any(r["result"].get("status") != "ok" for r in records):
            raise ValueError("This inference procedure requires every planned evaluation outcome to be observed and valid")
    if any(isinstance(r, dict) and isinstance(r.get("result"), dict) and
           r["result"].get("provider") == "human" for r in records):
        raise ValueError("Model records must not contain human providers; use human_path")
    model = _prepare(records, "model", return_tolerance_pp)
    if inference_method != "descriptive" and (model["exclusions"] or len(records) != len(expected)
                                               or any(i["metrics"] is None for i in model["candidates"])):
        raise ValueError("Inference requires a single structurally valid observation for every planned task")
    return_levels = _return_claim_levels(model)
    contrasts = _select_contrasts(model["candidates"])
    model_sets = _contrast_sets(model["buckets"], contrasts, "model", model["exclusions"])
    effects = _effects(model, model_sets, contrasts, bootstrap, seed)
    interaction_tables = _interactions(model, model_sets, contrasts, bootstrap, seed, model["exclusions"])
    coverage = _coverage(model, model_sets, contrasts)
    scenario_effects, scenario_robustness = _scenario_tables(model, model_sets, contrasts)
    template_effects = _template_effects(model, model_sets, contrasts, bootstrap, seed)
    validity = _validity_rows(model, model_sets, contrasts, bootstrap, seed)
    bounds = _bounds_rows(model, model_sets, contrasts)
    missingness = list(model["missingness"])
    exclusions = list(model["exclusions"])
    warnings = set(model["warnings"])
    design_summary = {"model": _design_summary(model)}
    comparisons, human_participant_rows, heterogeneity, human_flow = [], [], [], []
    administration = None
    record_totals = {"model": len(records)}
    resolved_design = None
    if human_path is not None:
        human_records = load_records(human_path)
        if roster is not None:
            validate_roster(human_records, roster)
        human_records = [r for r in human_records if not isinstance(r.get("trial"), dict)
                         or r["trial"].get("split") not in ("train", "dev", "test")
                         or r["trial"].get("split") in selected_splits]
        record_totals["human"] = len(human_records)
        if any(not isinstance(r, dict) or not isinstance(r.get("result"), dict) or
               r["result"].get("provider") != "human" for r in human_records):
            raise ValueError("human_path must contain imported human result records only")
        human = _prepare(human_records, "human", return_tolerance_pp)
        if assignment_path:
            if roster is None:
                raise ValueError("Assignment-aware analysis requires the original --trials roster")
            from .protocol import assignment_flow, load_assignments
            study = load_assignments(assignment_path, trials_path)
            assignment_flow(study, human_records, set())
            if any(bucket["rejected"] for bucket in human["buckets"]):
                raise ValueError("Assigned-study records contain structurally rejected observations")
        return_levels.extend(_return_claim_levels(human))
        # A subject/trial observation may appear only once even if a caller relabels
        # its human provider model/version/generation metadata.
        identifiers = [(r["result"].get("subject_id"), r["trial"].get("trial_id"))
                       for r in human_records if isinstance(r.get("trial"), dict)
                       and isinstance(r["result"].get("subject_id"), str)
                       and isinstance(r["trial"].get("trial_id"), str)]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Duplicate human (subject_id, trial_id) observations; comparisons would pseudoreplicate")
        resolved_design = _resolve_human_design(human["buckets"], human_design)
        human_sets = _human_sets(human["buckets"], resolved_design, "human", human["exclusions"])
        if assignment_path:
            from .protocol import administration_summary, assignment_flow, load_assignments
            if roster is None:
                raise ValueError("Assignment-aware analysis requires the original --trials roster")
            study = load_assignments(assignment_path, trials_path)
            if participants_path:
                participant_document = strict_json(Path(participants_path).read_text(encoding="utf-8"))
                provenance = participant_document.get("provenance", {})
                if (provenance.get("assignment_manifest_sha256") != study["manifest_sha256"]
                        or not provenance.get("instrument_sha256")):
                    raise ValueError("Assigned-study covariates require the same assignment manifest and documented instrument")
            analysed_ids = set()
            for pairs in human_sets["complete"][PRIMARY_CONTRAST].values():
                for pair in pairs:
                    if resolved_design == "within":
                        analysed_ids.update((pair["subject"], pair[side]["trial_id"]) for side in ("a", "b"))
                    else:
                        for side, responders in (("a", "event_resp"), ("b", "neutral_resp")):
                            analysed_ids.update((subject, pair[side]["trial_id"]) for subject, _ in pair[responders])
            normalized_records = [{"trial": item["trial"], "result": item["result"]} for item in human["candidates"]]
            human_flow = assignment_flow(study, normalized_records, analysed_ids,
                                         {t["trial_id"] for t in roster if t["split"] in selected_splits})
            administration = administration_summary(normalized_records)
            human_bounds, human_validity = _human_assignment_diagnostics(study, human, roster, selected_splits)
            bounds.extend(human_bounds)
            validity.extend(human_validity)
        else:
            warnings.add("Human assignment provenance is unverified; assigned/nonresponse counts are unknown and estimates are descriptive.")
        human_contrasts = [PRIMARY_CONTRAST]
        coverage.extend(_coverage(human, human_sets, human_contrasts, unpaired=resolved_design == "unpaired"))
        missingness.extend(human["missingness"])
        exclusions.extend(human["exclusions"])
        warnings.update(human["warnings"])
        effects.extend(_effects(human, human_sets, human_contrasts, bootstrap, seed))
        if resolved_design == "within":
            human_interactions = _interactions(human, human_sets, human_contrasts, bootstrap, seed, exclusions)
            for name, rows in human_interactions.items():
                interaction_tables[name].extend(rows)
        human_scenario_effects, human_scenario_robustness = _scenario_tables(human, human_sets, human_contrasts)
        scenario_effects.extend(human_scenario_effects)
        scenario_robustness.extend(human_scenario_robustness)
        validity.extend(_validity_rows(human, human_sets, human_contrasts, bootstrap, seed,
                                       contrast_effects=False))
        comparisons = _human_comparisons(model_sets["complete"].get(PRIMARY_CONTRAST, {}),
                                         human_sets["complete"][PRIMARY_CONTRAST], resolved_design,
                                         0, seed, exclusions)
        for row in comparisons:
            row.update(ci_unavailable_reason="descriptive_observed_participants_only",
                       inference_scope="descriptive_observed_participants_only")
        design_summary["human"] = _design_summary(human)
        if participants is not None:
            heterogeneity, observed = _heterogeneity(human["buckets"], resolved_design, participants,
                                                     bootstrap, seed, exclusions)
            human_participant_rows = _participant_tables(participants, observed)
    for summary in design_summary.values():
        warnings.update(summary["warnings"])
    if exclusions:
        warnings.add("Exclusions are explicit; consult exclusions and pair_coverage before interpreting complete-case effects.")
    if any(row["ci_unavailable_reason"] == "requires_two_subjects_per_arm_stratum" for row in comparisons):
        warnings.add("Human comparison intervals require at least two independent subjects in every "
                     "observed arm-membership stratum; singleton strata retain point estimates only.")
    tables_for_inference = {"effects": effects, **interaction_tables}
    preregistration = _apply_multiplicity(tables_for_inference, prereg)
    if preregistration["unmatched_primary"]:
        warnings.add(f"PREREGISTRATION: primary entries matched no analysis row: {', '.join(preregistration['unmatched_primary'])}")
    if preregistration["n_unavailable_primary_tests"]:
        warnings.add(f"PREREGISTRATION: {preregistration['n_unavailable_primary_tests']} primary test(s) "
                     "have no p-value; they remain in the Holm family as non-rejections.")
    status_totals = []
    for source, denominator in record_totals.items():
        counts = Counter()
        for row in missingness:
            if row["source"] == source:
                counts[row["status"]] += row["count"]
        counts["malformed"] += denominator - sum(counts.values())
        for status in (*STATUSES, "malformed"):
            status_totals.append({"source": source, "status": status,
                                  "count": counts[status], "denominator": denominator,
                                  "fraction": counts[status] / denominator if denominator else None})
    report = {
        "schema_version": 3, "bootstrap": bootstrap, "seed": seed, "confidence_level": None,
        "inference_method": inference_method, "evaluation_splits": sorted(selected_splits),
        "provenance": {"roster_verified": roster is not None,
                       "planned_evaluation_trials": sum(t["split"] in selected_splits for t in roster) if roster else None,
                       "observed_model_records": len(records),
                       "collection_protocol_verified": inference_method != "descriptive"},
        "return_tolerance_pp": return_tolerance_pp, "return_claim_levels": return_levels,
        "contrasts": contrasts, "primary_contrast": PRIMARY_CONTRAST, "human_design": resolved_design,
        "effects": effects, **interaction_tables, "validity_effects": validity,
        "bounds_effects": bounds, "scenario_robustness": scenario_robustness,
        "template_effects_rows": len(template_effects), "scenario_effect_rows": len(scenario_effects),
        "human_comparisons": comparisons, "human_participants": human_participant_rows,
        "human_heterogeneity": heterogeneity,
        "human_assignment_flow": human_flow,
        "human_administration": administration,
        "research_scope": {
            "status": "exploratory_prompt_sensitivity",
            "controls": "Existing controls are robustness comparisons with wording, norm, length and occasion confounds.",
            "languages": "AI-drafted translations; language plus wording until independent bilingual review and comprehension checks.",
            "calendar": "Named/explicit cues do not test unprompted date recognition; one avoidance occasion cannot separate valence from identity.",
            "outcomes": "Recommendation sensitivity, factual compliance, constraint compliance and hypothetical preferences are separate outcomes; no belief or irrationality composite.",
            "subgroups": "Observed surviving task mixtures; descriptive, not standardized covariate interactions.",
            "external_validation_required": ["independently authored matched controls and translations", "date verification and calendar-recognition checks",
                                             "independent scenario sampling and design-specific power/precision", "human recruitment, masking and comprehension pilot",
                                             "lexically disjoint cues and polarity counterfactuals", "expanded validated capability items"]},
        "preregistration": preregistration, "design_summary": design_summary,
        "missingness": missingness, "pair_coverage": coverage, "exclusions": exclusions,
        "record_status_totals": status_totals,
        "warnings": sorted(warnings),
        "methods": {
            "claim": "Estimates describe calendar-cue sensitivity: the change in a particular model's (or human sample's) answers when specified prompt text changes, under the stated inference conditions. They do not measure beliefs, and they do not identify mediating mechanisms or causal pathways inside the model.",
            "contrasts": "Effects compare specified prompt manipulations. Event-minus-control contrasts are robustness checks, not unique identification of auspiciousness, language, beliefs, preferences, or irrationality.",
            "estimand": "Average replicates/paraphrases within template, templates equally within subject/scenario, subjects equally within scenario, scenarios equally. Descriptive complete cases retain only surviving templates; prospective inference requires all planned outcomes valid.",
            "derived_metrics": "Signed return deviation = expected_return_pct - supplied gold return; return_claim_absolute_error_pp is its absolute value; return_claim_violation is 1 when absolute error exceeds report.return_tolerance_pp, allowing numerical rounding of 1e-12 at the boundary. Reserve shortfall = max(0, 100*liquidity_need_inr/capital_inr - cash_pct), with reserve_violation = 1 above 0.01 pp. wtp_premium_pct benchmarks against the quoted unit price. cash_pct and the signed return-deviation contrast are excluded from multiplicity families as redundant; related outcomes are not independent confirmations.",
            "return_claim_levels": "Per-condition signed return deviation, absolute error and violation rate among valid responses in structurally accepted buckets. Average repetitions within subject/scenario, subjects within scenario, and scenarios equally. These descriptive levels expose errors shared by both arms; they are not additional hypothesis tests. The tolerance is loaded from preregistration when declared, otherwise defaults to 0.01 pp or an explicit exploratory override; conflicting registered and command-line values are rejected.",
            "missingness": "Bounds include every structurally accepted pair with either condition observed; absent records and recorded nonvalid outcomes are separate. Entirely absent pairs need an external planned roster.",
            "validity_effects": "Model validity uses every accepted pair with either side observed. Human assignment validity uses all assigned responses, with nonresponse invalid in the submission/validity diagnostic only.",
            "bounds_effects": "Worst-case outcome-support bounds include absent and nonvalid records. Human assignment bounds average over all assigned subjects per task/arm, then templates/tasks within scenario and scenarios equally. Tasks never assigned one arm are flagged and excluded; bounds are not confidence intervals.",
            "scenario_robustness": "Per-scenario effects, sign agreement, range and per-template effects describe sampled situations; crossed prompts add repetitions, not independently sampled scenarios.",
            "effects_ci": "Population confidence intervals are suppressed. The scenario threshold is an eligibility rule, not a power justification.",
            "inference": "No ordinary-bootstrap tail p-values or population intervals. Optional scenario sign-flip tests require a collection-bound externally attested plan, approved original roster, immutable provider identities, full valid evaluation data and explicit independent sign symmetry. Exact through 16 scenarios, otherwise add-one Monte Carlo; suppressed below 12. Fixed primary identities/counts enter Holm with unavailable tests retained. No MME/equivalence claims from unadjusted intervals.",
            "language_interactions": "Task labels, economics and material hashes must agree across the compared dimensions; mismatches are excluded. A shared material hash does not certify translation equivalence. Human estimates remain descriptive.",
            "human_comparison": "Exact-task model-minus-human contrasts describe observed participants only; no participant-population interval or p-value. Human subgroup estimates use surviving mixtures and do not identify standardized interactions.",
            "participants": "Optional self-reported covariates (survey language, language proficiency, researcher-defined financial literacy score, observance). human_heterogeneity.csv splits the human event_minus_neutral effect (same estimator as the chosen design) by observance (>=4 vs <=2, middle excluded and counted) and by financial literacy (<= vs > the sample median of subjects with responses); it is descriptive and is never used to adjust the model comparison. Subjects absent from the participants file are listed in exclusions.",
            "human_unpaired_interval_eligibility": "No human-population interval is established for either design; observed participant/sample contrasts are descriptive.",
            "model_identity": "Provider, requested model, returned model version and generation settings are separate strata; unresolved versions cannot prove cross-run checkpoint identity.",
            "units": UNITS,
            "inference_scope": "Human and subgroup estimates are descriptive and never receive population p-values or adjusted significance.",
        },
    }
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stat = ("estimate", "ci_low", "ci_high", *INFERENCE_FIELDS)
    interaction_columns = (*GROUP_FIELDS, "source", "dimension", "reference_level", "level", "metric", "unit",
                           "contrast", "interaction", *stat, "matched_pairs", "n_clusters", "unmatched_tasks",
                           "unmatched_reference_tasks", "ambiguous_task_matches", "ci_method", "ci_unavailable_reason")
    tables = {
        "effects": (effects, (*GROUP_FIELDS, "source", "metric", "unit", "contrast", *stat, "n_pairs", "n_clusters", "n_subjects", "ci_method", "ci_unavailable_reason")),
        **{name: (interaction_tables[name], interaction_columns) for name in MULTIPLICITY_TABLES},
        "validity_effects": (validity, (*GROUP_FIELDS, "source", "row_type", "dimension", "level", "contrast", "estimate", "ci_low", "ci_high", "n_pairs", "n_clusters", "n_records", "n_ok", "n_invalid", "n_blocked", "n_error", "valid_rate", "ci_method", "ci_unavailable_reason")),
        "bounds_effects": (bounds, (*GROUP_FIELDS, "source", "metric", "unit", "contrast", "estimate_complete_case", "bound_low", "bound_high", "bounds_exclude_zero", "n_pairs_attempted", "n_pairs_complete", "share_missing_one_side", "n_absent_records", "n_recorded_nonvalid", "n_clusters_attempted", "n_tasks_missing_assigned_arm", "estimand", "skipped_reason")),
        "scenario_effects": (scenario_effects, (*GROUP_FIELDS, "source", "metric", "unit", "contrast", "cluster_id", "estimate", "n_pairs")),
        "scenario_robustness": (scenario_robustness, (*GROUP_FIELDS, "source", "metric", "unit", "contrast", "n_scenarios", "pooled_estimate", "share_same_sign", "min_scenario_estimate", "max_scenario_estimate")),
        "template_effects": (template_effects, (*GROUP_FIELDS, "source", "template_id", "metric", "unit", "contrast", "estimate", "ci_low", "ci_high", "n_pairs", "n_clusters", "ci_method", "ci_unavailable_reason")),
        "return_claim_levels": (return_levels, (*GROUP_FIELDS, "source", "condition", "metric", "unit",
                                               "estimate", "n_records", "n_clusters", "n_subjects")),
        "human_comparisons": (comparisons, (*GROUP_FIELDS, "metric", "unit", "contrast", "human_design", "estimate", "ci_low", "ci_high", "matched_model_pairs", "matched_human_units", "n_clusters", "n_subjects", "bootstrap_valid_draws", "ci_method", "ci_unavailable_reason", "unmatched_model_pairs")),
        "human_participants": (human_participant_rows, ("survey_language", "n_participants", "n_with_responses", "language_proficiency_mean", "language_proficiency_min", "language_proficiency_max", "financial_literacy_mean", "financial_literacy_median", "financial_literacy_min", "financial_literacy_max", "observance_mean", "observance_high_n", "observance_middle_n", "observance_low_n")),
        "human_assignment_flow": (human_flow, ("study_id", "language", "condition", "assigned", "submitted", "valid", "analysed", "nonresponse")),
        "human_heterogeneity": (heterogeneity, ("covariate", "level", "survey_language", "split_rule", "split_value", "metric", "unit", "contrast", "human_design", "estimate", "ci_low", "ci_high", "n_subjects_in_split", "n_subjects_contributing", "n_pairs", "n_clusters", "n_middle_excluded", "ci_method", "ci_unavailable_reason")),
        "missingness": (missingness, (*GROUP_FIELDS, "source", "status", "count", "denominator", "fraction")),
        "record_status_totals": (status_totals, ("source", "status", "count", "denominator", "fraction")),
        "pair_coverage": (coverage, (*GROUP_FIELDS, "source", "contrast", "observed_pairs", "attempted_pairs", "complete_pairs", "excluded_pairs", "complete_pair_coverage", "n_clusters")),
        "exclusions": (exclusions, ("source", "record_index", "pair_id", "contrast", "reason", "detail")),
    }
    report["outputs"] = {name: str(directory / f"{name}.csv") for name in tables}
    report["outputs"]["analysis"] = str(directory / "analysis.json")
    for name, (rows, fields) in tables.items():
        _csv_write(directory / f"{name}.csv", rows, fields)
    _json_write(directory / "analysis.json", report)
    if preregistration["unmatched_primary"]:
        raise ValueError("Preregistered primary entries matched no analysis row: "
                         f"{', '.join(preregistration['unmatched_primary'])}. Outputs were still written to {directory} "
                         "with the entries listed under preregistration.unmatched_primary.")
    return report


def import_humans(trials_path, csv_path, output_path, *, assignment_path=None, allow_unassigned=False):
    """Import consented, deidentified observations without inventing missing data.

    CSV booleans are true/false (case insensitive) or 1/0. Invalid answer cells
    become invalid result records; unknown trial IDs and duplicate observations
    are structural errors and abort the import before any output is written.
    Each subject may answer one trial of a pair (between-subject / counterbalanced
    designs, analysed with the unpaired estimator) or both (within design).
    """
    trials = read_jsonl(trials_path)
    validate_trials(trials)
    by_id = {trial["trial_id"]: trial for trial in trials}
    if assignment_path is None and not allow_unassigned:
        raise ValueError("Human import requires --assignments; legacy data require explicit --allow-unassigned")
    from .protocol import load_assignments
    study = load_assignments(assignment_path, trials_path) if assignment_path else None
    rows = _csv_read(csv_path, (*HUMAN_FIELDS, "study_id", "position") if study else HUMAN_FIELDS)
    seen = set()
    records = []
    status_counts = Counter()
    for line, row in rows:
        trial_id, subject_id = row["trial_id"].strip(), row["subject_id"].strip()
        if trial_id not in by_id:
            raise ValueError(f"{csv_path}:{line}: unknown trial_id {trial_id!r}")
        if not subject_id:
            raise ValueError(f"{csv_path}:{line}: subject_id must not be empty")
        assigned = None
        if study:
            assigned = study["index"].get((subject_id, trial_id))
            if (assigned is None or row["study_id"].strip() != study["study_id"] or
                    row["position"].strip() != assigned["position"] or
                    assigned["language"] != by_id[trial_id]["language"] or
                    assigned["condition"] != by_id[trial_id]["condition"] or
                    assigned["prompt_sha256"] != by_id[trial_id]["prompt_sha256"]):
                raise ValueError(f"{csv_path}:{line}: response does not match its study assignment")
        if (subject_id, trial_id) in seen:
            raise ValueError(f"{csv_path}:{line}: duplicate (subject_id, trial_id)")
        seen.add((subject_id, trial_id))
        raw_cells = {field: row[field] for field in PAYLOAD_FIELDS}
        result = {"provider": "human", "requested_model": "human", "model_version": "self_report",
                  "generation": {}, "usage": {}, "attempts": 1, "subject_id": subject_id,
                  "status": "ok", "payload": None, "raw_text": _canonical(raw_cells), "error": None}
        result["assignment"] = ({**assigned, "manifest_sha256": study["manifest_sha256"]} if study else
                                {"status": "unverified_legacy_assignment"})
        if row.get("submitted_at"):
            from datetime import datetime
            submitted = datetime.fromisoformat(row["submitted_at"].replace("Z", "+00:00"))
            if submitted.tzinfo is None:
                raise ValueError("submitted_at must include a timezone")
            result["submitted_at"] = submitted.isoformat()
        try:
            answer = {field: float(row[field]) for field in PAYLOAD_METRICS if field not in ("buy_now", "timing_days")}
            flag = row["buy_now"].strip().lower()
            if flag not in ("true", "false", "1", "0"):
                raise ValueError("buy_now must be true/false or 1/0")
            answer["buy_now"] = flag in ("true", "1")
            answer["timing_days"] = int(row["timing_days"].strip())
            answer["rationale"] = row["rationale"]
            result["raw_text"] = _canonical(answer)
            result = normalize_result(result, trial=by_id[trial_id])
        except (ValueError, TypeError, OverflowError) as exc:
            result.update(status="invalid", payload=None, error=str(exc))
        result["source_row"] = line
        result["source_cells"] = raw_cells
        status_counts[result["status"]] += 1
        records.append({"trial": by_id[trial_id], "result": result})
    if study:
        from .protocol import administration_summary
        administration_summary(records)
    write_jsonl(output_path, records)
    return {"records": len(records), "subjects": len({subject for subject, _ in seen}),
            "study_id": study["study_id"] if study else None,
            "assignment_verified": study is not None,
            "status_counts": {status: status_counts[status] for status in STATUSES},
            "output": str(output_path), "warnings": (["Invalid answers retained as invalid, never imputed."] if status_counts["invalid"] else []) +
            (["Legacy assignment provenance is unverified; descriptive use only."] if not study else [])}


def contextual_series(csv_path, output_dir):
    """Import sourced observational series without equating them to decisions.

    Each (series, unit, source) remains separate. Changes are absolute changes in
    the supplied unit, not automatically returns, allocations, or event effects.
    """
    rows = []
    seen = set()
    for line, row in _csv_read(csv_path, CONTEXT_FIELDS):
        item = {field: row[field].strip() for field in CONTEXT_FIELDS}
        try:
            parsed = date.fromisoformat(item["date"])
            if parsed.isoformat() != item["date"]:
                raise ValueError("date must use YYYY-MM-DD")
            value = float(item["value"])
            if not math.isfinite(value):
                raise ValueError("value must be finite")
            for field in ("series", "unit", "source"):
                if not item[field]:
                    raise ValueError(f"{field} must not be empty")
            key = tuple(item[field] for field in ("date", "series", "unit", "source"))
            if key in seen:
                raise ValueError("duplicate (date, series, unit, source) observation")
            seen.add(key)
        except ValueError as exc:
            raise ValueError(f"{csv_path}:{line}: {exc}") from exc
        item["value"] = value
        rows.append(item)
    rows.sort(key=lambda row: tuple(row[field] for field in ("series", "unit", "source", "date")))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["series"], row["unit"], row["source"])].append(row)
    summaries = []
    for (series, unit, source), observations in sorted(groups.items()):
        values = [row["value"] for row in observations]
        summaries.append({"series": series, "unit": unit, "source": source,
                          "n_observations": len(values), "first_date": observations[0]["date"],
                          "last_date": observations[-1]["date"], "first_value": values[0],
                          "last_value": values[-1], "minimum": min(values), "maximum": max(values),
                          "mean": fmean(values), "absolute_change": values[-1] - values[0] if len(values) >= 2 else None})
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "observations": len(rows), "series": summaries,
              "interpretation": "Descriptive contextual observations only. Event labels are supplied annotations, not verified dates or causal treatment assignments. Market prices, quantities, imports and returns are not individual allocations, preferences or decisions. No pooling across units/sources, market-human comparison, event study, currency conversion, interpolation or implied returns is performed.",
              "outputs": {"data": str(directory / "contextual_data.csv"),
                          "summary": str(directory / "contextual_summary.csv"),
                          "json": str(directory / "contextual_summary.json")}}
    _csv_write(directory / "contextual_data.csv", rows, CONTEXT_FIELDS)
    _csv_write(directory / "contextual_summary.csv", summaries,
               ("series", "unit", "source", "n_observations", "first_date", "last_date", "first_value", "last_value", "minimum", "maximum", "mean", "absolute_change"))
    _json_write(directory / "contextual_summary.json", report)
    return report
