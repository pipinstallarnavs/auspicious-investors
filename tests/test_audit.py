"""Audit regressions on synthetic fixtures; no research observations or real approvals."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from auspicious.analysis import HUMAN_FIELDS, analyze, import_humans, import_participants
from auspicious.core import read_jsonl, run_trials, write_jsonl
from auspicious.inference import ASSUMPTION, estimate, inference_scope, sign_flip_p
from auspicious.protocol import administration_summary, collection_protocol, immutable_provider, load_assignments
from auspicious.survey import assign_participants
from test_analysis import pair, payload
from test_runtime import CountingProvider
from test_survey import make_corpus
import test_stimuli


class InferenceCalibrationTests(unittest.TestCase):
    def test_exact_p_depends_on_observations_and_not_bootstrap_draws(self):
        self.assertEqual(sign_flip_p([1, 2], permutations=199), 0.5)
        self.assertEqual(sign_flip_p([1, 2], permutations=19999), 0.5)
        self.assertEqual(sign_flip_p([1] * 12), 2 / 4096)
        self.assertEqual(sign_flip_p([0] * 12), 1)

    def test_default_descriptive_and_small_cluster_suppression(self):
        self.assertEqual(estimate([1] * 12), (1, None, None, None))
        with inference_scope("scenario_sign_flip"):
            self.assertIsNone(estimate([1] * 11)[3])
            self.assertEqual(estimate([1] * 12)[3], 2 / 4096)
            self.assertIsNone(estimate([1] * 12, inferential=False)[3])
        self.assertIsNone(estimate([1] * 12)[3])

    def test_exact_sign_flip_null_calibration(self):
        rng = random.Random(20261024)
        rejections = sum(sign_flip_p([rng.gauss(0, 1) for _ in range(12)]) <= .05 for _ in range(1000))
        # Design-specific null diagnostic, not evidence that authored scenarios satisfy symmetry.
        self.assertLessEqual(rejections, 75)

    def test_monte_carlo_is_seeded_and_never_zero(self):
        result = sign_flip_p([1] * 17, seed=91, permutations=99)
        self.assertGreaterEqual(result, 1 / 100)
        self.assertEqual(result, sign_flip_p([1] * 17, seed=91, permutations=99))


class AuditPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def analyze(self, rows, **kwargs):
        path = self.root / "records.jsonl"
        write_jsonl(path, rows)
        return analyze(path, self.root / "analysis", **kwargs)

    def test_interaction_rejects_economic_and_material_mismatches(self):
        for field, value in (("economics", {**pair()[0]["trial"]["economics"], "gold_expected_annual_pct": 40}),
                             ("data_sha256", "different-materials")):
            rows = pair("s1", language="en") + pair("s1", language="hi")
            for row in rows[2:]:
                row["trial"][field] = value
            report = self.analyze(rows)
            row = next(r for r in report["language_interactions"] if r["metric"] == "gold_pct")
            self.assertEqual(row["matched_pairs"], 0)
            self.assertTrue(any(e["reason"] == "language_material_metadata_mismatch" for e in report["exclusions"]))

    def test_roster_rejects_relabelled_results_and_prompt_edits(self):
        rows = pair()
        roster = self.root / "trials.jsonl"
        write_jsonl(roster, [r["trial"] for r in rows])
        rows[0]["trial"]["economics"]["gold_expected_annual_pct"] = 40
        with self.assertRaisesRegex(ValueError, "original trial roster"):
            self.analyze(rows, trials_path=roster)
        rows = pair()
        rows[0]["trial"]["prompt"] += " edited"
        report = self.analyze(rows)
        self.assertTrue(report["exclusions"])
        self.assertTrue(all(r["estimate"] is None for r in report["effects"]))

    def test_default_split_and_explicit_exploratory_strata(self):
        rows = pair("s1", 10) + pair("s2", 20)
        for r in rows[:2]:
            r["trial"]["split"] = "train"
        report = self.analyze(rows)
        gold = [r for r in report["effects"] if r["metric"] == "gold_pct"]
        self.assertEqual([(r["split"], r["estimate"]) for r in gold], [("test", 20)])
        report = self.analyze(rows, evaluation_splits=("train", "test"))
        self.assertEqual({r["split"] for r in report["effects"]}, {"train", "test"})

    def test_equal_template_weights_despite_replication_imbalance(self):
        rows = pair("s1", 10, template="t1") + pair("s1", 10, template="t1", replicate=1) + pair("s1", 30, template="t2")
        report = self.analyze(rows)
        self.assertEqual(next(r for r in report["effects"] if r["metric"] == "gold_pct")["estimate"], 20)

    def test_unjustified_inference_fails_before_output(self):
        with self.assertRaisesRegex(ValueError, "sign-symmetry"):
            self.analyze(pair(), inference_method="scenario_sign_flip")
        self.assertFalse((self.root / "analysis").exists())

    def test_collection_binds_source_and_protocol_and_cache_invalidates(self):
        roster = self.root / "trials.jsonl"
        write_jsonl(roster, [r["trial"] for r in pair()])
        plan = self.root / "plan.json"
        plan.write_bytes(b'{\r\n"research_status":"exploratory"\r\n}\r\n')
        provider = CountingProvider()
        run_trials(roster, self.root / "run", provider, prereg_path=plan)
        records = read_jsonl(self.root / "run" / "records.jsonl")
        self.assertEqual(records[0]["collection"]["protocol_sha256"], hashlib.sha256(plan.read_bytes()).hexdigest())
        self.assertEqual((self.root / "run" / "preregistration.json").read_bytes(), plan.read_bytes())
        runtime = records[0]["collection"]["runtime"]
        self.assertIn("core.py", runtime["source_files"])
        self.assertIn("python", runtime)
        with patch("auspicious.protocol.runtime_identity", return_value={**runtime, "source_sha256": "changed"}):
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                run_trials(roster, self.root / "run", provider, prereg_path=plan)
        self.assertEqual(provider.calls, 2)

    def test_confirmatory_protocol_rejects_retrospective_or_unspecified_identity(self):
        plan = self.root / "plan.json"
        plan.write_text(json.dumps({"schema_version": 1, "primary": [{"id": "x", "table": "effects", "metric": "gold_pct",
             "contrast": "event_minus_neutral", "minimum_meaningful_effect": 2, "expected_tests": 1}]}))
        with self.assertRaisesRegex(ValueError, "external registration"):
            collection_protocol(plan, confirmatory=True)
        self.assertFalse(immutable_provider({"resolved_revision": "main", "tokenizer_resolved_revision": "main"}))
        self.assertTrue(immutable_provider({"resolved_revision": "a" * 40, "tokenizer_resolved_revision": "b" * 40}))
        self.assertFalse(immutable_provider({"resolved_revision": "a" * 40}))

    def test_prospectively_bound_synthetic_collection_and_tamper_rejection(self):
        fixture = test_stimuli.StimuliTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config.update(mode="confirmatory", languages=["en"], festivals=["dhanteras"], roles=["investor"],
                              scenarios=["home_deposit"], personas=["unspecified"])
        fixture.approve()
        fixture.build()
        filters = {"source": "model", "provider": "test-only", "requested_model": "fixture", "model_version": "None",
                   "generation_settings": '{"temperature":0}', "language": "en", "festival": "dhanteras",
                   "cue": "festival_named", "role": "investor", "persona_id": "unspecified", "split": "test"}
        document = {"schema_version": 1, "inference_method": "scenario_sign_flip", "inference_assumption": ASSUMPTION,
                    "registration": {"url": "https://example.invalid/SYNTHETIC-UNIT-TEST", "registered_at": "2000-01-01T00:00:00Z",
                                     "timestamp_verified_by": "SYNTHETIC TEST ONLY"}, "target_population": "synthetic unit fixtures",
                    "evaluation_splits": ["test"], "template_weighting": "equal_planned_templates_complete_case",
                    "primary": [{"id": "fixture", "table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
                                 "minimum_meaningful_effect": 2, "expected_tests": 1, "filters": filters}]}
        plan = self.root / "plan.json"
        plan.write_text(json.dumps(document))
        provider = CountingProvider()
        provider.identity["local_model_sha256"] = "f" * 64  # synthetic identity; no remote call
        run_trials(fixture.output_path, self.root / "run", provider, confirmatory=True, prereg_path=plan)
        records = read_jsonl(self.root / "run" / "records.jsonl")
        report = self.analyze(records, prereg_path=plan, trials_path=fixture.output_path)
        self.assertTrue(report["provenance"]["collection_protocol_verified"])
        self.assertIsNone(report["effects"][0]["p_value"])  # one scenario is still inadequate
        records[0]["collection"]["provider_identity"]["requested_model"] = "edited-provider"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.analyze(records, prereg_path=plan, trials_path=fixture.output_path)
        records = read_jsonl(self.root / "run" / "records.jsonl")
        records[0]["collection"]["protocol_sha256"] = "tampered"
        with self.assertRaisesRegex(ValueError, "before collection"):
            self.analyze(records, prereg_path=plan, trials_path=fixture.output_path)
        document["primary"][0]["filters"].pop("festival")
        plan.write_text(json.dumps(document))
        with self.assertRaisesRegex(ValueError, "hypothesis identity"):
            collection_protocol(plan, confirmatory=True)


class HumanAssignmentAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trials = make_corpus(languages=("en",), clusters=("one",), roles=("investor",), conditions=("event", "neutral"))
        for trial in self.trials:
            trial["split"] = "test"
        self.roster = self.root / "trials.jsonl"
        write_jsonl(self.roster, self.trials)
        self.study_dir = self.root / "study"
        self.report = assign_participants(self.roster, self.study_dir, participants=8, design="between",
                         items_per_participant=1, filler_fraction=0, study_id="audit-study",
                         assigned_at="2000-01-01T00:00:00Z", seed=9)
        self.manifest = self.study_dir / "design.json"
        self.study = load_assignments(self.manifest, self.roster)

    def response_csv(self, rows):
        path = self.root / "responses.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*HUMAN_FIELDS, "study_id", "position", "submitted_at"))
            writer.writeheader()
            for assigned in rows:
                writer.writerow({**payload(), "trial_id": assigned["trial_id"], "subject_id": assigned["subject_id"],
                                 "study_id": assigned["study_id"], "position": assigned["position"],
                                 "submitted_at": "2000-01-02T00:00:00Z"})
        return path

    def test_default_rejects_unassigned_unknown_subject_and_wrong_position(self):
        row = self.study["rows"][0]
        path = self.response_csv([row])
        with self.assertRaisesRegex(ValueError, "requires --assignments"):
            import_humans(self.roster, path, self.root / "out.jsonl")
        for bad in ({**row, "subject_id": "unknown"}, {**row, "position": "999"}):
            path = self.response_csv([bad])
            with self.assertRaisesRegex(ValueError, "study assignment"):
                import_humans(self.roster, path, self.root / "out.jsonl", assignment_path=self.manifest)

    def test_assignment_hash_and_study_scoping(self):
        self.assertTrue(all(r["subject_id"].startswith("audit-study-") for r in self.study["rows"]))
        target = self.study_dir / "assignments.csv"
        target.write_bytes(target.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_assignments(self.manifest)

    def test_flow_and_bounds_include_participants_with_no_submissions(self):
        event = next(r for r in self.study["rows"] if r["condition"] == "event")
        neutral = next(r for r in self.study["rows"] if r["condition"] == "neutral")
        human_path = self.root / "humans.jsonl"
        import_humans(self.roster, self.response_csv([event, neutral]), human_path, assignment_path=self.manifest)
        model_path = self.root / "model.jsonl"
        rows = [{"trial": t, "result": {"provider": "fixture", "requested_model": "fixture", "model_version": "1",
                 "generation": {}, "status": "ok", "payload": payload()}} for t in self.trials]
        write_jsonl(model_path, rows)
        report = analyze(model_path, self.root / "analysis", human_path=human_path,
                         trials_path=self.roster, assignment_path=self.manifest)
        flow = report["human_assignment_flow"]
        self.assertEqual(sum(r["assigned"] for r in flow), 8)
        self.assertEqual(sum(r["submitted"] for r in flow), 2)
        self.assertEqual(sum(r["nonresponse"] for r in flow), 6)
        self.assertEqual(sum(r["analysed"] for r in flow), 2)
        bound = next(r for r in report["bounds_effects"] if r["source"] == "human_assignment" and r["metric"] == "gold_pct")
        self.assertEqual((bound["bound_low"], bound["bound_high"]), (-75, 75))
        self.assertEqual(bound["n_absent_records"], 6)
        self.assertTrue(report["human_administration"]["order_verified"])
        imported = read_jsonl(human_path)
        imported[0]["result"]["assignment"]["position"] = "99"
        write_jsonl(human_path, imported)
        with self.assertRaisesRegex(ValueError, "altered"):
            analyze(model_path, self.root / "bad", human_path=human_path, trials_path=self.roster, assignment_path=self.manifest)

    def test_randomization_has_more_than_the_two_alternating_sequences(self):
        schedules = set()
        for seed in range(8):
            target = self.root / f"seed{seed}"
            assign_participants(self.roster, target, participants=8, design="between", items_per_participant=1,
                                filler_fraction=0, study_id="random-test", seed=seed)
            schedules.add(tuple(r["condition"] for r in load_assignments(target / "design.json")["rows"]))
        self.assertGreater(len(schedules), 2)
        self.assertFalse(self.report["preflight"]["field_ready"])

    def test_roster_requires_language_eligibility(self):
        path = self.root / "eligible.csv"
        path.write_text("subject_id,language,language_eligible\n" + "\n".join(f"p{i},en,false" for i in range(8)))
        with self.assertRaisesRegex(ValueError, "eligible"):
            assign_participants(self.roster, self.root / "eligible", participants=8, design="between",
                                items_per_participant=1, filler_fraction=0, participant_roster=path)

    def test_administration_order_is_checked_without_claiming_comprehension(self):
        record = {"result": {"subject_id": "one", "submitted_at": "2000-01-02T00:00:00Z",
                            "assignment": {"position": "1", "assigned_at": "2000-01-01T00:00:00Z"}}}
        other = copy.deepcopy(record)
        other["result"]["assignment"]["position"] = "2"
        other["result"]["submitted_at"] = "2000-01-01T12:00:00Z"
        self.assertEqual(administration_summary([record, other])["subjects_with_order_violations"], ["one"])
        other["result"]["submitted_at"] = "1999-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "precedes"):
            administration_summary([other])

    def test_covariates_bind_study_and_documented_instrument(self):
        row = self.study["rows"][0]
        path = self.root / "covariates.csv"
        path.write_text("subject_id,survey_language,language_proficiency,financial_literacy,observance_self_report\n"
                        + row["subject_id"] + ",en,5,8,3\n")
        instrument = self.root / "instrument.json"
        instrument.write_text(json.dumps({"id": "synthetic", "version": "1", "source": "unit fixture", "scoring": "0-10"}))
        output = self.root / "covariates.json"
        import_participants(path, output, assignment_path=self.manifest, instrument_path=instrument)
        provenance = json.loads(output.read_text())["provenance"]
        self.assertEqual(provenance["assignment_manifest_sha256"], self.study["manifest_sha256"])
        self.assertEqual(provenance["instrument_sha256"], hashlib.sha256(instrument.read_bytes()).hexdigest())
