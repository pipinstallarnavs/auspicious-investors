"""Synthetic fixtures only: a small corpus built in-test, no shipped stimuli, no human data."""
import csv
import hashlib
import json
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path

from auspicious.core import write_jsonl
from auspicious.survey import (
    ASSIGNMENT_FIELDS, PARTICIPANT_TEMPLATE_FIELDS, SURVEY_FIELDS, assign_participants,
)

CLUSTERS = tuple(f"scn_{name}" for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta"))
ECONOMICS = {"capital_inr": 100000, "gold_unit_price_inr": 10000, "transaction_fee_inr": 100,
             "horizon_years": 2, "liquidity_need_inr": 20000, "liquidity_due_days": 30,
             "gold_expected_annual_pct": 4, "equity_expected_annual_pct": 8, "cash_expected_annual_pct": 3,
             "gold_annual_volatility_pct": 15, "equity_annual_volatility_pct": 22,
             "cash_annual_volatility_pct": 0, "max_delay_days": 30}


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make_corpus(languages=("en", "hi"), clusters=CLUSTERS, roles=("investor", "advisor"),
                conditions=("event", "neutral", "no_calendar")):
    trials = []
    for language in languages:
        for index, cluster in enumerate(clusters):
            for role in roles:
                pair_id = sha(f"pair|{language}|{cluster}|{role}")
                for condition in conditions:
                    prompt = f"Question {sha(f'{pair_id}|{condition}')[:12]} about a household allocation ({language})."
                    trials.append({
                        "trial_id": sha(f"trial|{pair_id}|{condition}"), "pair_id": pair_id,
                        "cluster_id": cluster, "scenario_id": cluster, "split": ("train", "dev", "test")[index % 3],
                        "language": language, "festival": "dhanteras", "condition": condition,
                        "event_valence": "auspicious", "cue": "festival_named", "role": role,
                        "persona_id": "unspecified", "paraphrase_id": "direct", "template_id": "t1",
                        "replicate": 0, "calendar_pair_id": None, "calendar_pair": None,
                        "economics": ECONOMICS, "config_sha256": "0" * 64, "data_sha256": "1" * 64,
                        "calendar_sha256": None, "review_status": "draft", "synthetic": True,
                        "prompt": prompt, "prompt_sha256": sha(prompt),
                        "review": {"context_sha256": "2" * 64, "translation_status": "unreviewed_draft",
                                   "calendar_status": "hypothetical_not_calendar_dated"},
                    })
    return trials


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class SurveyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.counter = 0

    def corpus(self, **kwargs):
        trials = make_corpus(**kwargs)
        self.counter += 1
        path = self.directory / f"trials{self.counter}.jsonl"
        write_jsonl(path, trials)
        self.trials = {trial["trial_id"]: trial for trial in trials}
        return path

    def run_assign(self, path, name="out", **kwargs):
        kwargs.setdefault("study_id", "fixture")
        kwargs.setdefault("assigned_at", "2020-01-01T00:00:00+00:00")
        kwargs.setdefault("participants", 40)
        kwargs.setdefault("design", "between")
        kwargs.setdefault("items_per_participant", 4)
        kwargs.setdefault("filler_fraction", 0.25)
        output = self.directory / name
        report = assign_participants(path, output, **kwargs)
        return output, report

    def by_subject(self, output):
        grouped = defaultdict(list)
        for row in read_csv(output / "assignments.csv"):
            grouped[row["subject_id"]].append(row)
        return grouped

    def test_hard_invariants_both_designs(self):
        path = self.corpus(conditions=("event", "neutral", "no_calendar", "occasion_control"))
        for design in ("between", "counterbalanced"):
            with self.subTest(design=design):
                output, report = self.run_assign(path, name=design, design=design)
                grouped = self.by_subject(output)
                self.assertEqual(sorted(grouped), [f"fixture-S{n:04d}" for n in range(1, 41)])
                self.assertEqual(report["parameters"]["n_target_items"], 3)
                self.assertEqual(report["parameters"]["n_filler_items"], 1)
                for subject, rows in grouped.items():
                    self.assertEqual([int(r["position"]) for r in rows], [1, 2, 3, 4])
                    self.assertEqual(len({r["language"] for r in rows}), 1)
                    self.assertEqual(len({r["pair_id"] for r in rows}), 4)
                    self.assertEqual(len({r["cluster_id"] for r in rows}), 4)
                    for row in rows:
                        trial = self.trials[row["trial_id"]]
                        for field in ("condition", "cluster_id", "pair_id", "language"):
                            self.assertEqual(row[field], trial[field])
                    targets = [r for r in rows if r["kind"] == "target"]
                    fillers = [r for r in rows if r["kind"] == "filler"]
                    self.assertEqual((len(targets), len(fillers)), (3, 1))
                    self.assertTrue(all(r["condition"] in ("event", "neutral") for r in targets))
                    self.assertTrue(all(r["condition"] == "no_calendar" for r in fillers))
                    if design == "between":
                        self.assertEqual(len({r["condition"] for r in targets}), 1)
                    else:
                        counts = Counter(r["condition"] for r in targets)
                        self.assertLessEqual(abs(counts["event"] - counts["neutral"]), 1)
                expected = {"one_language_per_participant", "no_repeated_pair_id_within_participant",
                            "no_repeated_cluster_id_within_participant", "target_conditions_event_or_neutral_only",
                            "fillers_are_no_calendar_from_unused_clusters", "positions_complete"}
                if design == "between":
                    expected.add("single_arm_per_participant")
                self.assertEqual(set(report["invariants_checked"]), expected)
                self.assertTrue(all(report["invariants_checked"].values()))
                # Other controls are never assigned.
                self.assertNotIn("occasion_control", {r["condition"] for rows in grouped.values() for r in rows})

    def test_fillers_interleaved_at_random_positions(self):
        path = self.corpus()
        output, report = self.run_assign(path)
        positions = Counter(int(r["position"]) for rows in self.by_subject(output).values()
                            for r in rows if r["kind"] == "filler")
        self.assertGreater(len(positions), 1)
        self.assertEqual(sum(positions.values()), 40)
        distribution = report["balance"]["per_position_condition_counts"]
        self.assertEqual(sorted(distribution), ["1", "2", "3", "4"])
        for counts in distribution.values():
            self.assertEqual(sum(counts.values()), 40)
        self.assertEqual(sum(c.get("no_calendar", 0) for c in distribution.values()), 40)
        self.assertEqual(report["parameters"]["filler_fraction_realized"], 0.25)

    def test_language_assignment_balanced_and_filter(self):
        path = self.corpus(languages=("en", "hi", "ta"))
        output, report = self.run_assign(path, participants=10, items_per_participant=4)
        counts = Counter(rows[0]["language"] for rows in self.by_subject(output).values())
        self.assertEqual(sorted(counts.values()), [3, 3, 4])
        self.assertEqual(sum(info["participants"] for info in report["balance"]["per_language"].values()), 10)
        output, report = self.run_assign(path, name="filtered", participants=7, languages=["hi", "ta"])
        counts = Counter(rows[0]["language"] for rows in self.by_subject(output).values())
        self.assertEqual(set(counts), {"hi", "ta"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(report["parameters"]["languages"], ["hi", "ta"])
        for row in read_csv(output / "assignments.csv"):
            self.assertIn(row["language"], {"hi", "ta"})

    def test_between_arms_balanced_per_language(self):
        path = self.corpus(languages=("en", "hi", "ta"))
        output, report = self.run_assign(path, participants=31, design="between")
        arms = defaultdict(Counter)
        for rows in self.by_subject(output).values():
            targets = {r["condition"] for r in rows if r["kind"] == "target"}
            self.assertEqual(len(targets), 1)
            arms[rows[0]["language"]][targets.pop()] += 1
        for language, counts in arms.items():
            self.assertLessEqual(abs(counts["event"] - counts["neutral"]), 1, language)
            self.assertEqual(report["balance"]["per_language"][language]["arm_participants"],
                             {k: v for k, v in sorted(counts.items())})

    def test_counterbalanced_exposure_balance(self):
        path = self.corpus()
        output, report = self.run_assign(path, design="counterbalanced", participants=40)
        exposure = Counter()
        cluster_exposure = Counter()
        cluster_condition = Counter()
        condition_total = defaultdict(Counter)
        for rows in self.by_subject(output).values():
            for row in rows:
                if row["kind"] != "target":
                    continue
                trial = self.trials[row["trial_id"]]
                exposure[row["trial_id"]] += 1
                cluster_exposure[row["language"], row["cluster_id"]] += 1
                cluster_condition[row["language"], row["cluster_id"], row["condition"]] += 1
                condition_total[row["language"]][row["condition"]] += 1
        for language in ("en", "hi"):
            # 20 participants x 3 distinct clusters over 6 clusters: equal to the unit.
            spread = [cluster_exposure[language, c] for c in CLUSTERS]
            self.assertLessEqual(max(spread) - min(spread), 1)
            self.assertEqual(sum(spread), 60)
            self.assertEqual(spread, [10] * 6)
            self.assertLessEqual(abs(condition_total[language]["event"] - condition_total[language]["neutral"]), 1)
            info = report["balance"]["per_language"][language]
            self.assertEqual(info["cluster_target_exposure"], {"min": 10, "max": 10, "n_clusters": 6})
            for cluster in CLUSTERS:
                # Within one cluster/condition pool the greedy min-count rule keeps pairs within one exposure.
                for condition in ("event", "neutral"):
                    pool = [exposure[t["trial_id"]] for t in self.trials.values()
                            if t["language"] == language and t["cluster_id"] == cluster and t["condition"] == condition]
                    self.assertLessEqual(max(pool) - min(pool), 1)
                self.assertLessEqual(abs(cluster_condition[language, cluster, "event"]
                                         - cluster_condition[language, cluster, "neutral"]), 3)
            diffs = []
            for pair_id in {t["pair_id"] for t in self.trials.values() if t["language"] == language}:
                pair = {t["condition"]: t["trial_id"] for t in self.trials.values() if t["pair_id"] == pair_id}
                diffs.append(abs(exposure[pair["event"]] - exposure[pair["neutral"]]))
            self.assertEqual(info["pair_event_neutral_abs_diff_max"], max(diffs))
            self.assertLessEqual(max(diffs), 3)
        target_counts = [exposure[t["trial_id"]] for t in self.trials.values()
                         if t["condition"] in ("event", "neutral")]
        summary = report["balance"]["target_trial_exposure"]
        self.assertEqual((summary["min"], summary["max"], summary["n_trials"]),
                         (min(target_counts), max(target_counts), len(target_counts)))
        self.assertEqual(summary["unexposed"], target_counts.count(0))

    def test_filler_exposure_reported_and_balanced(self):
        path = self.corpus()
        output, report = self.run_assign(path, participants=40)
        counts = Counter()
        cluster_counts = Counter()
        for rows in self.by_subject(output).values():
            for row in rows:
                if row["kind"] == "filler":
                    counts[row["trial_id"]] += 1
                    cluster_counts[row["language"], row["cluster_id"]] += 1
        self.assertEqual(sum(counts.values()), 40)
        for language in ("en", "hi"):
            self.assertEqual(sum(cluster_counts[language, c] for c in CLUSTERS), 20)
        summary = report["balance"]["filler_trial_exposure"]
        self.assertEqual(summary["max"], max(counts.values()))

    def test_deterministic_given_seed(self):
        path = self.corpus()
        for design in ("between", "counterbalanced"):
            first, _ = self.run_assign(path, name=f"a-{design}", design=design, seed=7)
            second, _ = self.run_assign(path, name=f"b-{design}", design=design, seed=7)
            other, _ = self.run_assign(path, name=f"c-{design}", design=design, seed=8)
            files_a = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
            files_b = sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
            self.assertEqual(files_a, files_b)
            for relative in files_a:
                self.assertEqual((first / relative).read_bytes(), (second / relative).read_bytes(), str(relative))
            self.assertNotEqual((first / "assignments.csv").read_bytes(), (other / "assignments.csv").read_bytes())

    def test_survey_files_do_not_leak_metadata(self):
        path = self.corpus(conditions=("event", "neutral", "no_calendar", "purchase_control"))
        output, _ = self.run_assign(path, design="counterbalanced", participants=12)
        survey_files = sorted((output / "surveys").glob("*.csv"))
        self.assertEqual([p.stem for p in survey_files], [f"fixture-S{n:04d}" for n in range(1, 13)])
        assignments = self.by_subject(output)
        forbidden = [*CLUSTERS, "dhanteras", "no_calendar", "purchase_control", "occasion_control",
                     "event", "neutral", "filler", "target", "kind", "condition", "cluster", "pair_id"]
        forbidden += [trial["pair_id"] for trial in self.trials.values()]
        for survey in survey_files:
            text = survey.read_text(encoding="utf-8")
            self.assertEqual(text.splitlines()[0], ",".join(SURVEY_FIELDS))
            for needle in forbidden:
                self.assertNotIn(needle, text, f"{survey.name} leaks {needle}")
            rows = read_csv(survey)
            expected = assignments[survey.stem]
            self.assertEqual([r["position"] for r in rows], [r["position"] for r in expected])
            self.assertEqual([r["trial_id"] for r in rows], [r["trial_id"] for r in expected])
            for row in rows:
                self.assertEqual(row["prompt"], self.trials[row["trial_id"]]["prompt"])
                self.assertEqual(set(row), set(SURVEY_FIELDS))

    def test_output_files_and_participant_template(self):
        path = self.corpus()
        output, report = self.run_assign(path, participants=6, seed=11)
        with open(output / "assignments.csv", encoding="utf-8", newline="") as stream:
            self.assertEqual(next(csv.reader(stream)), list(ASSIGNMENT_FIELDS))
        template = read_csv(output / "participants-template.csv")
        with open(output / "participants-template.csv", encoding="utf-8", newline="") as stream:
            self.assertEqual(next(csv.reader(stream)), list(PARTICIPANT_TEMPLATE_FIELDS))
        self.assertEqual([r["subject_id"] for r in template], [f"fixture-S{n:04d}" for n in range(1, 7)])
        grouped = self.by_subject(output)
        for row in template:
            self.assertEqual(row["survey_language"], grouped[row["subject_id"]][0]["language"])
            for field in PARTICIPANT_TEMPLATE_FIELDS[2:]:
                self.assertEqual(row[field], "")
        design = json.loads((output / "design.json").read_text(encoding="utf-8"))
        self.assertEqual(design["seed"], 11)
        self.assertEqual(design["design"], "between")
        self.assertEqual(design["trials"]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(design["parameters"]["participants"], 6)
        self.assertIn("unpaired", design["analysis_note"])
        self.assertEqual(design["invariants_checked"], report["invariants_checked"])
        self.assertEqual(report["output_dir"], str(output))

    def test_zero_filler_fraction_needs_no_no_calendar_trials(self):
        path = self.corpus(conditions=("event", "neutral"))
        output, report = self.run_assign(path, filler_fraction=0, items_per_participant=3, participants=8)
        self.assertEqual(report["parameters"]["n_filler_items"], 0)
        self.assertIsNone(report["balance"]["filler_trial_exposure"])
        self.assertTrue(all(r["kind"] == "target" for r in read_csv(output / "assignments.csv")))

    def test_error_cases(self):
        path = self.corpus()
        no_filler_path = self.corpus(conditions=("event", "neutral"))
        cases = [
            ("no no_calendar trials", dict(trials=no_filler_path), "no_calendar"),
            ("bad design", dict(design="within"), "design"),
            ("zero participants", dict(participants=0), "participants"),
            ("float participants", dict(participants=2.5), "participants"),
            ("zero items", dict(items_per_participant=0), "items_per_participant"),
            ("too many items for clusters", dict(items_per_participant=7), "clusters"),
            ("filler fraction one", dict(filler_fraction=1), "filler_fraction"),
            ("negative filler fraction", dict(filler_fraction=-0.1), "filler_fraction"),
            ("no target items", dict(items_per_participant=1, filler_fraction=0.5), "target"),
            ("unknown language", dict(languages=["fr"]), "Languages"),
            ("empty languages", dict(languages=[]), "languages"),
            ("bad seed", dict(seed="42"), "seed"),
        ]
        for label, overrides, message in cases:
            with self.subTest(label):
                trials = overrides.pop("trials", path)
                name = "err-" + label.replace(" ", "-")
                with self.assertRaisesRegex(ValueError, message):
                    self.run_assign(trials, name=name, **overrides)
                self.assertFalse((self.directory / name).exists())

    def test_fillers_need_enough_clusters(self):
        path = self.corpus(clusters=CLUSTERS[:4])
        # 4 clusters, 4 items: 3 targets + 1 filler fit; 4 items with 0.5 fillers (2) also fit.
        self.run_assign(path, name="fit", items_per_participant=4, filler_fraction=0.5, participants=4)
        with self.assertRaisesRegex(ValueError, "clusters"):
            self.run_assign(path, name="toomany", items_per_participant=5, participants=4)

    def test_refuses_non_empty_output_directory(self):
        path = self.corpus()
        output = self.directory / "occupied"
        output.mkdir()
        (output / "keep.txt").write_text("existing", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "non-empty"):
            assign_participants(path, output, participants=2, design="between", items_per_participant=4)
        self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "existing")
        self.assertEqual([p.name for p in output.iterdir()], ["keep.txt"])
        empty = self.directory / "empty"
        empty.mkdir()
        assign_participants(path, empty, participants=2, design="between", items_per_participant=4)
        self.assertTrue((empty / "design.json").exists())
        with self.assertRaisesRegex(ValueError, "non-empty"):
            assign_participants(path, empty, participants=2, design="between", items_per_participant=4)

    def test_invalid_trials_rejected(self):
        trials = make_corpus()
        trials = [t for t in trials if not (t["condition"] == "neutral" and t["pair_id"] == trials[0]["pair_id"])]
        path = self.directory / "broken.jsonl"
        write_jsonl(path, trials)
        with self.assertRaises(ValueError):
            self.run_assign(path, name="broken")


if __name__ == "__main__":
    unittest.main()
