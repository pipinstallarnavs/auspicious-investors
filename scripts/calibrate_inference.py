"""Synthetic sign-symmetry diagnostics, not empirical investment results.

Run from the repository root: python scripts/calibrate_inference.py --out report.json
These simulations cannot justify independence or symmetry for an authored corpus.
"""
import argparse
import json
import math
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auspicious.inference import MIN_SCENARIOS, sign_flip_p


def wilson(successes, total):
    z = 1.959963984540054
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def simulate(experiments=1000, seed=20261024, scenario_counts=(2, 4, 12), effect_sds=(0, 0.5)):
    rng = random.Random(seed)
    rows = []
    for count in scenario_counts:
        for effect in effect_sds:
            rejected = sum(sign_flip_p([rng.gauss(effect, 1) for _ in range(count)]) <= .05
                           for _ in range(experiments))
            rows.append({"n_independent_scenarios": count, "effect_in_scenario_sd": effect,
                         "experiments": experiments, "raw_test_rejections": rejected,
                         "raw_test_rejection_rate": rejected / experiments,
                         "monte_carlo_rate_wilson_95": wilson(rejected, experiments),
                         "pipeline_inference_eligible_by_cluster_count": count >= MIN_SCENARIOS,
                         "pipeline_rejections_if_other_protocol_gates_satisfied": rejected if count >= MIN_SCENARIOS else 0})
    return {"kind": "synthetic_inference_diagnostic", "seed": seed, "nominal_alpha": .05,
            "sampling_model": "independent Normal(effect_in_scenario_sd, 1) scenario effects",
            "test": "two-sided scenario mean sign flips; exact through 16; otherwise add-one Monte Carlo (9999)",
            "scope": "Conditional calibration/power under the stated simulated model only. Not evidence of adequate coverage, independence, symmetry, or power for the shipped authored tasks.",
            "small_sample_example": {"effects": [1, 2], "p_value": sign_flip_p([1, 2]), "pipeline_p_value": None},
            "rows": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--scenario-counts", type=int, nargs="+", default=[2, 4, 12])
    parser.add_argument("--effect-sds", type=float, nargs="+", default=[0, 0.5])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.experiments < 1 or any(n < 1 for n in args.scenario_counts) or any(not math.isfinite(x) for x in args.effect_sds):
        parser.error("Require positive experiment/scenario counts and finite effect sizes")
    result = simulate(args.experiments, args.seed, args.scenario_counts, args.effect_sds)
    Path(args.out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
