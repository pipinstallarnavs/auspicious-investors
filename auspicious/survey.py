"""Human-survey assignment for the between-subject / counterbalanced comparison.

A participant who sees both the event and the neutral version of a nearly
identical question can infer the purpose of the study. This module therefore
assigns each participant a set of trials such that:

* one language only (the language is chosen round-robin over participants);
* never two trials from the same ``pair_id`` and never two from the same
  ``cluster_id`` (scenario), so a participant never supplies both conditions
  of any pair, nor two answers about the same scenario;
* target items use the ``event`` / ``neutral`` conditions only;
* ``no_calendar`` trials of scenarios not used by that participant's targets are
  interleaved as fillers at random positions.

``design="between"``: each participant is placed in one arm (event or neutral,
balanced per language) and all target items come from that arm.
``design="counterbalanced"``: conditions vary across a participant's different
scenarios and are balanced across participants so each pair's event and neutral
trial are exposed about equally often.

Downstream, responses collected with these assignments are single-condition per
pair per subject, so they must be analysed with the *unpaired* human estimator
(``analysis`` with ``human_design="unpaired"``), not the within-subject one.

Assignment is greedy minimum-exposure with seeded tie-breaking and is fully
deterministic given the corpus and seed. Survey CSVs contain only position,
trial id, language and prompt text; condition/cluster/pair/festival metadata
stays in ``assignments.csv`` and ``design.json``.
"""
from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import random
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path

from .core import atomic_text, read_jsonl, validate_trials

DESIGNS = ("between", "counterbalanced")
TARGET_CONDITIONS = ("event", "neutral")
FILLER_CONDITION = "no_calendar"
ASSIGNMENT_FIELDS = ("subject_id", "position", "trial_id", "kind", "condition", "cluster_id", "pair_id", "language",
                     "study_id", "block_id", "session_id", "assigned_at", "prompt_sha256", "language_eligible")
SURVEY_FIELDS = ("position", "trial_id", "language", "prompt")
PARTICIPANT_TEMPLATE_FIELDS = (
    "subject_id", "survey_language", "language_proficiency", "financial_literacy",
    "observance_self_report", "age_band", "region",
)
ANALYSIS_NOTE = (
    "Each participant receives at most one trial per pair_id and per cluster_id, so no "
    "participant supplies both conditions of a pair. Analyse the collected responses with "
    "the unpaired (between-subject) human estimator (analysis human_design='unpaired'); "
    "the within-subject estimator does not apply. Filler (no_calendar) responses are masking "
    "items and are not part of the primary event-minus-neutral comparison. Participant "
    "covariates are collected separately (participants-template.csv -> import_participants); "
    "survey_language is prefilled from the assignment, all measured covariates are blank."
)


def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _check_arguments(participants, design, items, filler_fraction, seed):
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    if not _is_int(participants) or participants < 1:
        raise ValueError("participants must be a positive integer")
    if not _is_int(items) or items < 1:
        raise ValueError("items_per_participant must be a positive integer")
    if isinstance(filler_fraction, bool) or not isinstance(filler_fraction, (int, float)) \
            or not 0 <= filler_fraction < 1:
        raise ValueError("filler_fraction must be a number in [0, 1)")
    if not _is_int(seed):
        raise ValueError("seed must be an integer")


def _check_output_dir(directory):
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        raise ValueError(f"Refusing to write into non-empty output path: {directory}")


def _csv_text(header, rows):
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


class _State:
    """Exposure counters for one language."""

    def __init__(self):
        self.trial = Counter()
        self.cluster_target = Counter()
        self.cluster_condition = Counter()
        self.cluster_filler = Counter()
        self.condition_total = Counter()


def _index(trials, languages):
    targets = defaultdict(lambda: defaultdict(dict))  # language -> cluster -> pair -> condition -> trial
    fillers = defaultdict(lambda: defaultdict(list))  # language -> cluster -> [no_calendar trials]
    for trial in sorted(trials, key=lambda item: item["trial_id"]):
        language = trial["language"]
        if language not in languages:
            continue
        if trial["condition"] in TARGET_CONDITIONS:
            targets[language][trial["cluster_id"]].setdefault(trial["pair_id"], {})[trial["condition"]] = trial
        elif trial["condition"] == FILLER_CONDITION:
            fillers[language][trial["cluster_id"]].append(trial)
    return targets, fillers


def _target_conditions(design, chosen, state, arm, rng):
    if design == "between":
        return {cluster: arm for cluster in chosen}
    n_event = len(chosen) // 2
    if len(chosen) % 2:
        event_total, neutral_total = state.condition_total["event"], state.condition_total["neutral"]
        if event_total < neutral_total or (event_total == neutral_total and rng.random() < 0.5):
            n_event += 1
    # Clusters that have been shown neutral more often than event get event first.
    ranked = sorted(chosen, key=lambda c: (state.cluster_condition[c, "event"] - state.cluster_condition[c, "neutral"],
                                           rng.random()))
    event_clusters = set(ranked[:n_event])
    return {cluster: "event" if cluster in event_clusters else "neutral" for cluster in chosen}


def _pick_target(pairs, condition, state, rng):
    other = "neutral" if condition == "event" else "event"

    def key(pair_id):
        mine = state.trial[pairs[pair_id][condition]["trial_id"]]
        theirs = state.trial[pairs[pair_id][other]["trial_id"]]
        return (mine, mine - theirs, rng.random())

    return pairs[min(sorted(pairs), key=key)][condition]


def _participant_items(language, design, arm, n_targets, n_fillers, targets, fillers, state, rng):
    clusters = sorted(targets[language])
    order = sorted(clusters, key=lambda c: (state.cluster_target[c], rng.random()))
    chosen = order[:n_targets]
    conditions = _target_conditions(design, chosen, state, arm, rng)
    target_items = [(_pick_target(targets[language][c], conditions[c], state, rng), "target") for c in chosen]
    filler_items = []
    if n_fillers:
        taken = set(chosen)
        available = sorted((c for c in fillers[language] if c not in taken),
                           key=lambda c: (state.cluster_filler[c], rng.random()))
        if len(available) < n_fillers:
            raise ValueError(f"Not enough no_calendar clusters for fillers in language {language}")
        for cluster in available[:n_fillers]:
            trial = min(fillers[language][cluster], key=lambda t: (state.trial[t["trial_id"]], rng.random()))
            filler_items.append((trial, "filler"))
    rng.shuffle(target_items)
    rng.shuffle(filler_items)
    filler_positions = set(rng.sample(range(n_targets + n_fillers), n_fillers))
    target_iter, filler_iter = iter(target_items), iter(filler_items)
    items = [next(filler_iter) if position in filler_positions else next(target_iter)
             for position in range(n_targets + n_fillers)]
    for trial, kind in items:
        state.trial[trial["trial_id"]] += 1
        if kind == "target":
            state.cluster_target[trial["cluster_id"]] += 1
            state.cluster_condition[trial["cluster_id"], trial["condition"]] += 1
            state.condition_total[trial["condition"]] += 1
        else:
            state.cluster_filler[trial["cluster_id"]] += 1
    return items


def _verify(rows, design, n_targets, n_fillers, items):
    """Re-check the hard invariants on the final assignment rows."""
    by_subject = defaultdict(list)
    for row in rows:
        by_subject[row["subject_id"]].append(row)
    for subject, subject_rows in by_subject.items():
        if [row["position"] for row in subject_rows] != list(range(1, items + 1)):
            raise ValueError(f"{subject}: positions are not 1..{items}")
        if len({row["language"] for row in subject_rows}) != 1:
            raise ValueError(f"{subject}: more than one language")
        if len({row["pair_id"] for row in subject_rows}) != items:
            raise ValueError(f"{subject}: repeated pair_id")
        if len({row["cluster_id"] for row in subject_rows}) != items:
            raise ValueError(f"{subject}: repeated cluster_id")
        targets = [row for row in subject_rows if row["kind"] == "target"]
        fillers = [row for row in subject_rows if row["kind"] == "filler"]
        if len(targets) != n_targets or len(fillers) != n_fillers:
            raise ValueError(f"{subject}: wrong target/filler counts")
        if any(row["condition"] not in TARGET_CONDITIONS for row in targets):
            raise ValueError(f"{subject}: target with non event/neutral condition")
        if any(row["condition"] != FILLER_CONDITION for row in fillers):
            raise ValueError(f"{subject}: filler is not a no_calendar trial")
        if design == "between" and len({row["condition"] for row in targets}) != 1:
            raise ValueError(f"{subject}: mixed arms in between design")
    invariants = {
        "one_language_per_participant": True,
        "no_repeated_pair_id_within_participant": True,
        "no_repeated_cluster_id_within_participant": True,
        "target_conditions_event_or_neutral_only": True,
        "fillers_are_no_calendar_from_unused_clusters": True,
        "positions_complete": True,
    }
    if design == "between":
        invariants["single_arm_per_participant"] = True
    return invariants


def _exposure_summary(counts):
    if not counts:
        return None
    return {"min": min(counts), "max": max(counts), "mean": round(sum(counts) / len(counts), 6),
            "n_trials": len(counts), "unexposed": sum(1 for value in counts if value == 0)}


def _balance(languages, targets, fillers, states, participant_info, rows):
    per_language = {}
    all_target, all_filler = [], []
    for language in languages:
        state = states[language]
        target_counts, filler_counts, pair_diff, cluster_counts = [], [], 0, []
        for cluster, pairs in targets[language].items():
            cluster_counts.append(state.cluster_target[cluster])
            for pair in pairs.values():
                exposure = {c: state.trial[pair[c]["trial_id"]] for c in TARGET_CONDITIONS}
                target_counts.extend(exposure.values())
                pair_diff = max(pair_diff, abs(exposure["event"] - exposure["neutral"]))
        for trials in fillers[language].values():
            filler_counts.extend(state.trial[t["trial_id"]] for t in trials)
        all_target.extend(target_counts)
        all_filler.extend(filler_counts)
        subjects = [s for s, info in participant_info.items() if info["language"] == language]
        per_language[language] = {
            "participants": len(subjects),
            "arm_participants": dict(sorted(Counter(participant_info[s]["arm"] for s in subjects).items()))
            if any(participant_info[s]["arm"] for s in subjects) else None,
            "target_items_by_condition": {c: state.condition_total[c] for c in TARGET_CONDITIONS},
            "filler_items": sum(state.cluster_filler.values()),
            "target_trial_exposure": _exposure_summary(target_counts),
            "filler_trial_exposure": _exposure_summary(filler_counts),
            "cluster_target_exposure": {"min": min(cluster_counts), "max": max(cluster_counts),
                                        "n_clusters": len(cluster_counts)},
            "pair_event_neutral_abs_diff_max": pair_diff,
        }
    positions = defaultdict(Counter)
    for row in rows:
        positions[row["position"]][row["condition"]] += 1
    return {
        "per_language": per_language,
        "target_trial_exposure": _exposure_summary(all_target),
        "filler_trial_exposure": _exposure_summary(all_filler),
        "per_position_condition_counts": {str(p): dict(sorted(c.items())) for p, c in sorted(positions.items())},
        "note": "Exposure statistics cover every eligible trial of the used languages; trials never shown count as 0 "
                "(min exposure is 0 whenever the pool has more trials than the participants can cover).",
    }


def assign_participants(trials_path, output_dir, *, participants, design, items_per_participant,
                        filler_fraction=0.3, seed=42, languages=None, study_id=None,
                        participant_roster=None, assigned_at=None):
    """Write per-participant survey files plus the assignment table and design report.

    ``items_per_participant`` is the total number of items a participant sees
    (targets plus fillers); fillers = round-half-up(items * filler_fraction) and
    every item comes from a distinct scenario cluster, so it cannot exceed the
    number of clusters available in the participant's language.
    """
    _check_arguments(participants, design, items_per_participant, filler_fraction, seed)
    study_id = study_id or ("study-" + uuid.uuid4().hex[:16])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{2,63}", study_id):
        raise ValueError("study_id must be a safe unique study identifier")
    assigned_at = assigned_at or dt.datetime.now(dt.timezone.utc).isoformat()
    if dt.datetime.fromisoformat(assigned_at.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("Assignment timestamps must include a timezone")
    output = Path(output_dir)
    _check_output_dir(output)
    trials_path = Path(trials_path)
    trials = read_jsonl(trials_path)
    validate_trials(trials)
    corpus_languages = sorted({trial["language"] for trial in trials})
    if languages is None:
        used = corpus_languages
    else:
        if isinstance(languages, str) or not languages:
            raise ValueError("languages must be a non-empty list of language codes")
        unknown = sorted(set(languages) - set(corpus_languages))
        if unknown:
            raise ValueError(f"Languages not present in trials: {unknown}")
        used = sorted(set(languages))
    items = items_per_participant
    n_fillers = int(items * filler_fraction + 0.5)
    n_targets = items - n_fillers
    if n_targets < 1:
        raise ValueError("filler_fraction leaves no target items")
    targets, fillers = _index(trials, set(used))
    if n_fillers and not any(fillers[language] for language in used):
        raise ValueError("Fillers requested but the trials contain no no_calendar condition; "
                         "build with controls including no_calendar or set filler_fraction=0")
    for language in used:
        if len(targets[language]) < items:
            raise ValueError(f"Language {language}: {len(targets[language])} scenario clusters available, "
                             f"need items_per_participant={items} distinct clusters")
        if n_fillers and len(fillers[language]) < n_fillers:
            raise ValueError(f"Language {language}: {len(fillers[language])} no_calendar clusters, "
                             f"need {n_fillers} fillers")

    rng = random.Random(seed)
    roster = None
    if participant_roster is not None:
        with open(participant_roster, encoding="utf-8-sig", newline="") as stream:
            roster = list(csv.DictReader(stream))
        if len(roster) != participants or len({r.get("subject_id") for r in roster}) != participants:
            raise ValueError("Participant roster must contain the declared number of unique subjects")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", r.get("subject_id", "")) or r.get("language") not in used or
               r.get("language_eligible", "").lower() not in ("true", "1") for r in roster):
            raise ValueError("Every roster subject must be declared eligible in their survey language")
        rng.shuffle(roster)
    language_order = list(used)
    rng.shuffle(language_order)
    language_counts = Counter(r["language"] for r in roster) if roster else Counter(
        language_order[i % len(language_order)] for i in range(participants))
    arms, blocks = {}, {}
    for language in language_order:
        schedule, identifiers = [], []
        while len(schedule) < language_counts[language]:
            remaining = language_counts[language] - len(schedule)
            size = min(4, remaining)
            values = ["event", "neutral"] * (size // 2)
            if size % 2:
                values.append(rng.choice(TARGET_CONDITIONS))
            rng.shuffle(values)
            identifiers.extend([f"{language}-B{len(schedule) // 4 + 1:04d}"] * size)
            schedule.extend(values)
        arms[language], blocks[language] = schedule, identifiers
    language_position = Counter()
    states = {language: _State() for language in used}
    rows, surveys, participant_info = [], {}, {}
    for index in range(participants):
        person = roster[index] if roster else {}
        subject = study_id + "-" + person.get("subject_id", f"S{index + 1:04d}")
        language = person.get("language", language_order[index % len(language_order)])
        position = language_position[language]
        arm = arms[language][position] if design == "between" else None
        block = blocks[language][position]
        language_position[language] += 1
        participant_items = _participant_items(language, design, arm, n_targets, n_fillers,
                                               targets, fillers, states[language], rng)
        participant_info[subject] = {"language": language, "arm": arm, "block_id": block}
        for position, (trial, kind) in enumerate(participant_items, 1):
            rows.append({"subject_id": subject, "position": position, "trial_id": trial["trial_id"], "kind": kind,
                         "condition": trial["condition"], "cluster_id": trial["cluster_id"],
                         "pair_id": trial["pair_id"], "language": language, "study_id": study_id,
                         "block_id": block, "session_id": person.get("session_id", "draft_unspecified"),
                         "assigned_at": assigned_at, "prompt_sha256": trial["prompt_sha256"],
                         "language_eligible": bool(roster)})
        surveys[subject] = [[position, trial["trial_id"], language, trial["prompt"]]
                            for position, (trial, _kind) in enumerate(participant_items, 1)]
    invariants = _verify(rows, design, n_targets, n_fillers, items)

    trials_bytes = trials_path.read_bytes()
    design_report = {
        "schema_version": 2, "study_id": study_id, "assigned_at": assigned_at,
        "language_eligibility_verified": roster is not None,
        "participant_roster_sha256": hashlib.sha256(Path(participant_roster).read_bytes()).hexdigest() if roster else None,
        "randomization": "randomized permuted blocks within language; block size <=4; conceal master assignments and seed from enrolment staff",
        "status": "draft_assignment_not_yet_fielded",
        "seed": seed,
        "design": design,
        "parameters": {
            "participants": participants, "items_per_participant": items, "n_target_items": n_targets,
            "n_filler_items": n_fillers, "filler_fraction_requested": filler_fraction,
            "filler_fraction_realized": n_fillers / items, "languages": used,
            "language_assignment_order": language_order,
        },
        "trials": {"path": str(trials_path), "sha256": hashlib.sha256(trials_bytes).hexdigest(),
                   "n_trials_in_file": len(trials)},
        "analysis_note": ANALYSIS_NOTE,
        "balance": _balance(used, targets, fillers, states, participant_info, rows),
        "invariants_checked": invariants,
        "files": {"assignments": "assignments.csv", "surveys": "surveys/<subject_id>.csv",
                  "participants_template": "participants-template.csv"},
    }
    output.mkdir(parents=True, exist_ok=True)
    for subject, survey_rows in surveys.items():
        atomic_text(output / "surveys" / f"{subject}.csv", _csv_text(SURVEY_FIELDS, survey_rows))
    atomic_text(output / "assignments.csv",
                _csv_text(ASSIGNMENT_FIELDS, [[row[field] for field in ASSIGNMENT_FIELDS] for row in rows]))
    design_report["assignments_sha256"] = hashlib.sha256((output / "assignments.csv").read_bytes()).hexdigest()
    target_coverage = defaultdict(Counter)
    for row in rows:
        if row["kind"] == "target":
            target_coverage[row["pair_id"]][row["condition"]] += 1
    design_report["preflight"] = {
        "target_pairs_observed": len(target_coverage),
        "target_pairs_assigned_both_arms": sum(all(c[k] for k in TARGET_CONDITIONS) for c in target_coverage.values()),
        "target_pairs_missing_assigned_arm": sum(not all(c[k] for k in TARGET_CONDITIONS) for c in target_coverage.values()),
        "unexposed_target_trials": design_report["balance"]["target_trial_exposure"]["unexposed"],
        "field_ready": False,
        "required_before_fielding": ["independent language eligibility and comprehension checks", "masking/suspicion pilot",
                                     "review exact task/arm coverage", "freeze estimand, weighting, exclusions and stopping",
                                     "secure assignment concealment and record actual response timestamps"]}
    atomic_text(output / "participants-template.csv", _csv_text(
        PARTICIPANT_TEMPLATE_FIELDS,
        [[subject, info["language"], "", "", "", "", ""] for subject, info in participant_info.items()]))
    atomic_text(output / "design.json", _json_text(design_report))
    return {**design_report, "output_dir": str(output)}
