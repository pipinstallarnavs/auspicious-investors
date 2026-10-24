"""Command-line entry point. Optional model dependencies load only on demand."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

from .core import load_review_context, read_jsonl, run_trials, validate_trials, write_jsonl

SELECT_FIELDS = ("language", "festival", "cue", "role", "persona_id", "scenario_id",
                 "paraphrase_id", "template_id", "split")


def _add_model_options(command, *, temperature=0.0):
    command.add_argument("--model", required=True)
    command.add_argument("--revision")
    command.add_argument("--device", default="cpu")
    command.add_argument("--temperature", type=float, default=temperature)
    command.add_argument("--max-output-tokens", type=int, default=512)
    command.add_argument("--seed", type=int, default=42)


def _add_steering_options(command):
    command.add_argument("--steering-path")
    command.add_argument("--strength", type=float, default=0.0)
    command.add_argument("--layer", type=int)
    command.add_argument("--random-direction-seed", type=int)
    command.add_argument("--steering-dose", choices=("activation_std", "raw_coefficient"), default="activation_std")
    command.add_argument("--steering-positions", choices=("all", "prompt_last", "generated"), default="all",
                         help="Token positions that receive the steering vector")


def _open_provider(args):
    from .openweights import OpenWeightProvider
    return OpenWeightProvider(args.model, revision=args.revision, device=args.device,
                              temperature=args.temperature, max_output_tokens=args.max_output_tokens,
                              seed=args.seed, steering_path=args.steering_path, strength=args.strength,
                              layer=args.layer, random_direction_seed=args.random_direction_seed,
                              steering_positions=args.steering_positions, steering_dose=args.steering_dose)


def parser():
    root = argparse.ArgumentParser(prog="auspicious", description="Multilingual calendar-cue sensitivity experiments")
    commands = root.add_subparsers(dest="command", required=True)

    build = commands.add_parser("build", help="Build deterministic paired trials from a JSON design")
    build.add_argument("--config", required=True)
    build.add_argument("--out", required=True)
    review = commands.add_parser("review-export", help="Export hash-bound attestations for human review")
    review.add_argument("--config", required=True)
    review.add_argument("--out", required=True)
    validate = commands.add_parser("validate", help="Check pair integrity, hashes, and optional review approvals")
    validate.add_argument("--trials", required=True)
    validate.add_argument("--confirmatory", action="store_true")
    select = commands.add_parser("select", help="Create a whole-pair exploratory/probe subset")
    select.add_argument("--trials", required=True)
    select.add_argument("--out", required=True)
    for field in SELECT_FIELDS:
        select.add_argument("--" + field.replace("_", "-"), action="append")

    commands.add_parser("models", help="List Gemini text-generation models accessible to your key")
    run = commands.add_parser("run", help="Run/resume Gemini calls with SQLite checkpoints")
    local = commands.add_parser("run-openweights", help="Run/resume a local Hugging Face decoder-only model")
    for command in (run, local):
        command.add_argument("--trials", required=True)
        command.add_argument("--out", required=True)
        command.add_argument("--limit", type=int, help="Exploratory only; rounded down to whole pairs (all conditions)")
        command.add_argument("--confirmatory", action="store_true")
        command.add_argument("--prereg", help="Analysis plan bound before the first collected response")
        command.add_argument("--retry-errors", action="store_true",
                             help="Retry transport errors only; preserves invalid/blocked outputs")
    run.add_argument("--model", required=True)
    run.add_argument("--temperature", type=float, default=0.7)
    run.add_argument("--max-output-tokens", type=int, default=1024)
    run.add_argument("--seed", type=int, default=None)
    run.add_argument("--concurrency", type=int, default=2)
    run.add_argument("--rpm", type=float, default=30, help="Global request-start rate across all supplied keys")
    run.add_argument("--timeout", type=float, default=120)
    run.add_argument("--max-attempts", type=int, default=4)
    _add_model_options(local)
    _add_steering_options(local)

    analysis = commands.add_parser("analyze", help="Clustered effects, controls, interactions, validity and bounds")
    analysis.add_argument("--records", required=True)
    analysis.add_argument("--out", required=True)
    analysis.add_argument("--bootstrap", type=int, default=2000, help="Legacy option; does not enable population CIs or p-values")
    analysis.add_argument("--seed", type=int, default=42)
    analysis.add_argument("--humans", help="Validated human records JSONL from import-humans")
    analysis.add_argument("--prereg", help="Preregistration JSON declaring primary contrasts and minimum effects")
    analysis.add_argument("--trials", help="Original planned trial roster for provenance checks")
    analysis.add_argument("--split", dest="evaluation_splits", action="append", choices=("train", "dev", "test"),
                          help="Evaluation split; default test. Repeat only for explicitly exploratory pooling.")
    analysis.add_argument("--inference", choices=("descriptive", "scenario_sign_flip"), default=None)
    analysis.add_argument("--assignments", help="Original study design.json for human assignment validation")
    analysis.add_argument("--return-tolerance-pp", type=float,
                          help="Absolute return-error tolerance in percentage points (default 0.01 or prereg value)")
    analysis.add_argument("--human-design", choices=("auto", "within", "unpaired"), default="auto")
    analysis.add_argument("--participants", help="Validated participant covariates JSON from import-participants")
    human = commands.add_parser("import-humans", help="Validate observed same-task participant CSV responses")
    human.add_argument("--trials", required=True)
    human.add_argument("--csv", required=True)
    human.add_argument("--out", required=True)
    human.add_argument("--assignments", help="Study design.json binding subject, trial, arm, position and language")
    human.add_argument("--allow-unassigned", action="store_true", help="Explicit legacy descriptive-only import")
    participants = commands.add_parser("import-participants",
                                       help="Validate participant covariates (literacy, language, observance)")
    participants.add_argument("--csv", required=True)
    participants.add_argument("--out", required=True)
    participants.add_argument("--assignments", help="Original study design.json")
    participants.add_argument("--instrument", help="JSON documenting instrument id, version, source and scoring")
    assign = commands.add_parser("assign-participants",
                                 help="Between-participant/counterbalanced survey assignment with fillers")
    assign.add_argument("--trials", required=True)
    assign.add_argument("--out", required=True)
    assign.add_argument("--participants", type=int, required=True, help="Number of participants")
    assign.add_argument("--design", choices=("between", "counterbalanced"), required=True)
    assign.add_argument("--items-per-participant", type=int, required=True)
    assign.add_argument("--filler-fraction", type=float, default=0.3)
    assign.add_argument("--seed", type=int, default=42)
    assign.add_argument("--study-id", required=True)
    assign.add_argument("--participant-roster", help="Language-eligible subject/session roster CSV")
    assign.add_argument("--language", action="append", help="Restrict to a language; repeatable")
    context = commands.add_parser("context", help="Summarize sourced external market series descriptively")
    context.add_argument("--csv", required=True)
    context.add_argument("--out", required=True)

    extract = commands.add_parser("extract", help="Extract prompt-only final-token residual activations")
    extract.add_argument("--trials", required=True)
    extract.add_argument("--out", required=True)
    extract.add_argument("--model", required=True)
    extract.add_argument("--revision")
    extract.add_argument("--layer", type=int, default=-1)
    extract.add_argument("--device", default="cpu")
    probe = commands.add_parser("probe", help="Probe with text baseline, held-out transfer and random-direction null")
    probe.add_argument("--activations", required=True)
    probe.add_argument("--out", required=True)
    probe.add_argument("--ridge", type=float, default=1.0)
    capability = commands.add_parser("capability", help="Arithmetic/financial capability check (optionally steered)")
    capability.add_argument("--out", required=True)
    capability.add_argument("--items", help="Capability items JSON (default: data/capability.json)")
    capability.add_argument("--limit", type=int)
    _add_model_options(capability)
    _add_steering_options(capability)
    grid = commands.add_parser("steering-grid",
                               help="Layer x position x strength grid with random-direction controls and capability checks")
    grid.add_argument("--trials", required=True, help="Single-valence trials; direction train, behavior selection dev")
    grid.add_argument("--out", required=True)
    grid.add_argument("--layers", type=int, nargs="+", required=True, help="Layer indices, space-separated (negative allowed)")
    grid.add_argument("--strengths", type=float, nargs="+", required=True, help="Steering strengths, space-separated (negative allowed)")
    grid.add_argument("--positions", nargs="+", choices=("all", "prompt_last", "generated"), default=["all"])
    grid.add_argument("--random-seeds", type=int, nargs="*", default=[42], help="Seeds for activation-scale matched random controls")
    grid.add_argument("--evaluation-split", choices=("dev", "test"), default="dev")
    grid.add_argument("--frozen-selection", help="Frozen dev-selection JSON required for one-cell test evaluation")
    grid.add_argument("--ridge", type=float, default=1.0)
    grid.add_argument("--limit", type=int)
    grid.add_argument("--no-capability", action="store_true")
    _add_model_options(grid)
    freeze = commands.add_parser("freeze-steering", help="Freeze exactly one completed dev steering cell")
    freeze.add_argument("--grid", required=True)
    freeze.add_argument("--layer", required=True, type=int)
    freeze.add_argument("--position", required=True, choices=("all", "prompt_last", "generated"))
    freeze.add_argument("--strength", required=True, type=float)
    freeze.add_argument("--out", required=True)
    return root


def dispatch(args):
    if args.command == "build":
        from .stimuli import build_trials
        return build_trials(args.config, args.out)
    if args.command == "review-export":
        from .stimuli import export_review
        return export_review(args.config, args.out)
    if args.command == "validate":
        trials = read_jsonl(args.trials)
        validate_trials(trials, confirmatory=args.confirmatory,
                        review_context=load_review_context(args.trials, args.confirmatory))
        return {"valid": True, "trials": len(trials), "confirmatory": args.confirmatory,
                "review_statuses": sorted({t["review_status"] for t in trials}),
                "conditions": sorted({t["condition"] for t in trials})}
    if args.command == "select":
        trials = read_jsonl(args.trials)
        validate_trials(trials)
        selected = [trial for trial in trials if all(
            getattr(args, field) is None or trial[field] in getattr(args, field) for field in SELECT_FIELDS)]
        validate_trials(selected)
        write_jsonl(args.out, selected)
        sidecar = Path(str(args.trials) + ".review.json")
        if sidecar.exists():
            shutil.copyfile(sidecar, str(args.out) + ".review.json")
        return {"output": args.out, "trials": len(selected),
                "warning": "Subset is exploratory unless it is the entire reviewed design"}
    if args.command == "models":
        from .gemini import list_models
        return list_models()
    if args.command == "run":
        from .gemini import GeminiProvider
        provider = GeminiProvider(args.model, temperature=args.temperature,
                                  max_output_tokens=args.max_output_tokens, seed=args.seed,
                                  rpm=args.rpm, timeout=args.timeout, max_attempts=args.max_attempts)
        return run_trials(args.trials, args.out, provider, confirmatory=args.confirmatory,
                          concurrency=args.concurrency, limit=args.limit, retry_errors=args.retry_errors,
                          prereg_path=args.prereg)
    if args.command == "run-openweights":
        return run_trials(args.trials, args.out, _open_provider(args), confirmatory=args.confirmatory,
                          concurrency=1, limit=args.limit, retry_errors=args.retry_errors, prereg_path=args.prereg)
    if args.command == "analyze":
        from .analysis import analyze
        return analyze(args.records, args.out, bootstrap=args.bootstrap, seed=args.seed,
                       human_path=args.humans, prereg_path=args.prereg,
                       human_design=args.human_design, participants_path=args.participants,
                       return_tolerance_pp=args.return_tolerance_pp, trials_path=args.trials,
                       evaluation_splits=args.evaluation_splits or ("test",), inference_method=args.inference,
                       assignment_path=args.assignments)
    if args.command == "import-humans":
        from .analysis import import_humans
        return import_humans(args.trials, args.csv, args.out, assignment_path=args.assignments,
                             allow_unassigned=args.allow_unassigned)
    if args.command == "import-participants":
        from .analysis import import_participants
        return import_participants(args.csv, args.out, assignment_path=args.assignments, instrument_path=args.instrument)
    if args.command == "assign-participants":
        from .survey import assign_participants
        return assign_participants(args.trials, args.out, participants=args.participants, design=args.design,
                                   items_per_participant=args.items_per_participant,
                                   filler_fraction=args.filler_fraction, seed=args.seed,
                                   languages=args.language, study_id=args.study_id,
                                   participant_roster=args.participant_roster)
    if args.command == "context":
        from .analysis import contextual_series
        return contextual_series(args.csv, args.out)
    if args.command == "extract":
        from .openweights import extract_activations
        return extract_activations(args.trials, args.out, model=args.model, revision=args.revision,
                                   layer=args.layer, device=args.device)
    if args.command == "probe":
        from .openweights import fit_probe
        return fit_probe(args.activations, args.out, ridge=args.ridge)
    if args.command == "capability":
        from .openweights import capability_check
        return capability_check(args.out, _open_provider(args), items_path=args.items, limit=args.limit)
    if args.command == "steering-grid":
        from .openweights import steering_grid
        return steering_grid(args.trials, args.out, model=args.model, revision=args.revision, device=args.device,
                             layers=args.layers, strengths=args.strengths, positions=tuple(args.positions),
                             random_seeds=tuple(args.random_seeds), temperature=args.temperature,
                             max_output_tokens=args.max_output_tokens, seed=args.seed, ridge=args.ridge,
                             limit=args.limit, capability=not args.no_capability,
                             evaluation_split=args.evaluation_split, frozen_selection=args.frozen_selection)
    if args.command == "freeze-steering":
        from .openweights import freeze_steering_selection
        return freeze_steering_selection(args.grid, args.out, layer=args.layer, position=args.position, strength=args.strength)
    raise ValueError("Unknown command")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted; committed results are preserved. Resume with the same command.", file=sys.stderr)
        return 130
