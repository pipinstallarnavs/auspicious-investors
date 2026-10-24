"""Synthetic unit-test fixtures only: no claimed human or market observations."""
import copy
import csv
import hashlib
import json
import tempfile
from functools import partial
import unittest
from pathlib import Path

from auspicious.analysis import (
    CONTEXT_FIELDS, HUMAN_FIELDS, _adjust_family, _benjamini_hochberg, _holm, analyze, contextual_series,
    import_humans, import_participants,
)
from auspicious.core import write_jsonl

legacy_import_humans = partial(import_humans, allow_unassigned=True)

PARTICIPANT_FIELDS = ("subject_id", "survey_language", "language_proficiency", "financial_literacy",
                      "observance_self_report", "age_band", "region")
ECONOMICS = {
    "capital_inr": 200000, "gold_unit_price_inr": 10000, "transaction_fee_inr": 100,
    "horizon_years": 2, "liquidity_need_inr": 100000, "liquidity_due_days": 30,
    "gold_expected_annual_pct": 4, "equity_expected_annual_pct": 8, "cash_expected_annual_pct": 3,
    "gold_annual_volatility_pct": 15, "equity_annual_volatility_pct": 22,
    "cash_annual_volatility_pct": 0, "max_delay_days": 30,
}
# The liquidity reserve is 100000/200000 = 50% of capital, held entirely in cash.


def payload(gold=30, buy=False, equity=40, expected=4, wtp=5000):
    return {"gold_pct": gold, "equity_pct": equity, "cash_pct": 100 - gold - equity,
            "buy_now": buy, "timing_days": 0 if buy else 7,
            "willingness_to_pay_inr": wtp, "expected_return_pct": expected,
            "rationale": "Synthetic test answer."}


def pair(cluster="s1", difference=10, language="en", replicate=0, version="v1",
         subject=None, role="investor", calendar=None, persona="default",
         cue="festival_named", template="t1", golds=None, answers=None,
         conditions=("event", "neutral")):
    pair_id = f"{cluster}-{language}-{replicate}-{role}-{calendar}-{persona}-{cue}-{template}"
    golds = {"event": 30 + difference, "neutral": 30, **(golds or {})}
    records = []
    for condition in conditions:
        prompt = f"Synthetic fixture {pair_id} {condition}"
        trial = {"trial_id": f"{pair_id}-{condition}", "pair_id": pair_id,
                 "cluster_id": cluster, "scenario_id": cluster, "split": "test",
                 "language": language, "festival": "dhanteras", "condition": condition,
                 "event_valence": "auspicious", "cue": cue, "role": role,
                 "persona_id": persona, "paraphrase_id": "p0", "template_id": template,
                 "replicate": replicate, "calendar_pair_id": calendar,
                 "economics": dict(ECONOMICS), "prompt": prompt,
                 "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "review_status": "draft", "review": {}}
        answer = (answers or {}).get(condition) or payload(golds.get(condition, 30))
        result = {"status": "ok", "raw_text": json.dumps(answer), "payload": answer,
                  "provider": "fake", "requested_model": "fixture", "model_version": version,
                  "generation": {"temperature": 0}, "usage": {}, "attempts": 1, "error": None}
        if subject is not None:
            result.update(provider="human", requested_model="human", model_version="self_report",
                          generation={}, subject_id=subject)
        records.append({"trial": trial, "result": result})
    return records


def spoil(record, status="invalid"):
    record["result"].update(status=status, payload=None, raw_text="not an answer")
    return record


def human(records, condition, subject, gold):
    """One human response to ONE trial of a pair (between-subject / counterbalanced)."""
    record = copy.deepcopy(next(r for r in records if r["trial"]["condition"] == condition))
    answer = payload(gold)
    record["result"].update(provider="human", requested_model="human", model_version="self_report",
                            generation={}, subject_id=subject, payload=answer, raw_text=json.dumps(answer))
    return record


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def run_analysis(self, records, humans=None, bootstrap=100, seed=42, **kwargs):
        records_path = self.root / "records.jsonl"
        write_jsonl(records_path, records)
        if humans is not None:
            human_path = self.root / "humans.jsonl"
            write_jsonl(human_path, humans)
            kwargs["human_path"] = human_path
        return analyze(records_path, self.root / "analysis", bootstrap=bootstrap, seed=seed, **kwargs)

    def row(self, report, metric, table="effects", **filters):
        filters.setdefault("source", "model")
        if table == "effects":
            filters.setdefault("contrast", "event_minus_neutral")
        return next(r for r in report[table] if r["metric"] == metric and
                    all(r.get(key) == value for key, value in filters.items()))

    def gold(self, report, **filters):
        return self.row(report, "gold_pct", **filters)

    def csv_file(self, name, fields, rows):
        path = self.root / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def read_csv(self, name):
        with (self.root / "analysis" / f"{name}.csv").open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def write_prereg(self, document):
        path = self.root / "prereg.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    # ----- carried-over behaviour -----

    def test_equal_scenario_weight_not_repetition_weight(self):
        records = pair("s1", 10) + pair("s1", 10, replicate=1) + pair("s1", 10, replicate=2) + pair("s2", 0)
        row = self.gold(self.run_analysis(records))
        self.assertEqual(row["estimate"], 5)
        self.assertEqual(row["n_pairs"], 4)
        self.assertEqual(row["n_clusters"], 2)
        self.assertEqual(row["unit"], "percentage_points")
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(row["ci_high"])

    def test_seed_and_output_are_deterministic(self):
        records = pair("s1", 5) + pair("s2", 20) + pair("s3", -5)
        first = self.run_analysis(records)
        second = self.run_analysis(records)
        self.assertEqual(first, second)
        written = json.loads((self.root / "analysis" / "analysis.json").read_text())
        self.assertEqual(written, second)
        self.assertTrue((self.root / "analysis" / "effects.csv").read_text().startswith("provider,"))

    def test_single_cluster_has_no_inferential_interval(self):
        row = self.gold(self.run_analysis(pair()))
        self.assertEqual(row["estimate"], 10)
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(row["p_value"])
        self.assertEqual(row["ci_unavailable_reason"], "population_interval_not_established")

    def test_disabled_bootstrap(self):
        row = self.gold(self.run_analysis(pair("s1") + pair("s2"), bootstrap=0))
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["ci_unavailable_reason"], "population_interval_not_established")

    def test_invalid_blocked_error_and_missing_pairs_are_counted(self):
        records = pair("valid")
        for status in ("invalid", "blocked", "error"):
            bad = pair(status)
            spoil(bad[0], status)
            records.extend(bad)
        records.extend(pair("missing")[:1])
        report = self.run_analysis(records)
        self.assertEqual(self.gold(report)["n_pairs"], 1)
        coverage = report["pair_coverage"][0]
        self.assertEqual(coverage["contrast"], "event_minus_neutral")
        self.assertEqual(coverage["observed_pairs"], 5)
        self.assertEqual(coverage["attempted_pairs"], 5)
        self.assertEqual(coverage["complete_pair_coverage"], .2)
        counts = {row["status"]: row for row in report["missingness"]}
        self.assertEqual(counts["blocked"]["count"], 1)
        self.assertEqual(counts["invalid"]["denominator"], 9)
        self.assertTrue(report["warnings"])
        self.assertEqual(len(report["exclusions"]), 4)

    def test_ok_label_does_not_bypass_payload_validation(self):
        records = pair()
        records[0]["result"]["payload"]["gold_pct"] = 101
        report = self.run_analysis(records)
        self.assertIsNone(self.gold(report)["estimate"])
        self.assertEqual(next(r for r in report["missingness"] if r["status"] == "invalid")["count"], 1)

    def test_malformed_third_record_invalidates_pair_and_counts_globally(self):
        records = pair()
        malformed = copy.deepcopy(records[0])
        del malformed["trial"]["scenario_id"]
        records.extend([malformed, {"unexpected": "shape"}])
        report = self.run_analysis(records)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        total = next(row for row in report["record_status_totals"] if row["status"] == "malformed")
        self.assertEqual(total["count"], 2)
        self.assertEqual(total["denominator"], 4)

    def test_missing_economics_or_template_is_malformed(self):
        for field in ("economics", "template_id"):
            with self.subTest(field=field):
                records = pair()
                del records[0]["trial"][field]
                report = self.run_analysis(records)
                self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
                self.assertTrue(any("malformed_record" in r["reason"] for r in report["exclusions"]))

    def test_binary_outcome_uses_probability_difference_not_percentage(self):
        records = pair()
        records[0]["result"]["payload"].update(buy_now=True, timing_days=0)
        report = self.run_analysis(records)
        row = self.row(report, "buy_now")
        self.assertEqual(row["estimate"], 1)
        self.assertEqual(row["unit"], "probability_difference")

    def test_duplicate_trial_excludes_whole_pair(self):
        records = pair()
        records.append(copy.deepcopy(records[0]))
        report = self.run_analysis(records)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        self.assertTrue(any("duplicate_trial_observation" in row["reason"] for row in report["exclusions"]))

    def test_relabelled_duplicate_task_is_not_pseudoreplicated(self):
        records = pair()
        clone = copy.deepcopy(records)
        for record in clone:
            record["trial"]["pair_id"] += "-copy"
            record["trial"]["trial_id"] += "-copy"
        report = self.run_analysis(records + clone)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        self.assertTrue(any("duplicate_semantic_task" in r["reason"] for r in report["exclusions"]))

    def test_pair_metadata_mismatch_is_excluded(self):
        records = pair()
        records[1]["trial"]["scenario_id"] = "different"
        report = self.run_analysis(records)
        self.assertEqual(report["pair_coverage"][0]["complete_pairs"], 0)
        self.assertIn("inconsistent_pair_metadata", [r["reason"] for r in report["exclusions"]])
        records = pair()
        records[1]["trial"]["template_id"] = "t2"
        report = self.run_analysis(records)
        self.assertIn("inconsistent_pair_metadata", [r["reason"] for r in report["exclusions"]])

    def test_versions_and_settings_are_never_mixed(self):
        records = pair("s1", 10, version="v1") + pair("s1", -10, version="v2")
        report = self.run_analysis(records)
        self.assertEqual(self.gold(report, model_version="v1")["estimate"], 10)
        self.assertEqual(self.gold(report, model_version="v2")["estimate"], -10)
        records = pair()
        records[1]["result"]["model_version"] = "v2"
        report = self.run_analysis(records)
        self.assertTrue(all(r["complete_pairs"] == 0 for r in report["pair_coverage"]))
        records = pair()
        records[1]["result"]["generation"] = {"temperature": 1}
        report = self.run_analysis(records)
        self.assertTrue(all(r["complete_pairs"] == 0 for r in report["pair_coverage"]))

    def test_language_interaction_uses_identical_tasks_and_paired_clusters(self):
        records = (pair("s1", 5) + pair("s1", 15, language="hi") +
                   pair("s2", -5) + pair("s2", 5, language="hi") +
                   pair("s3", 25, language="hi"))
        report = self.run_analysis(records)
        row = self.row(report, "gold_pct", "language_interactions")
        self.assertEqual(row["estimate"], 10)
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(row["ci_high"])
        self.assertEqual(row["matched_pairs"], 2)
        self.assertEqual(row["unmatched_tasks"], 1)
        self.assertEqual((row["dimension"], row["reference_level"], row["level"]), ("language", "en", "hi"))
        self.assertEqual(report["persona_interactions"], [])
        self.assertEqual(report["cue_interactions"], [])

    def test_language_interaction_does_not_match_different_calendar_pairs(self):
        records = pair(calendar="a") + pair(language="hi", calendar="b")
        report = self.run_analysis(records)
        self.assertTrue(report["language_interactions"])
        self.assertTrue(all(row["matched_pairs"] == 0 for row in report["language_interactions"]))

    def test_language_roles_not_conflated(self):
        report = self.run_analysis(pair(role="investor") + pair(language="hi", role="advisor"))
        self.assertEqual(report["language_interactions"], [])

    def test_empty_input_produces_header_only_tables(self):
        report = self.run_analysis([])
        self.assertEqual(report["effects"], [])
        for name in ("effects", "human_comparisons", "missingness", "exclusions", "validity_effects",
                     "bounds_effects", "scenario_effects", "template_effects", "human_heterogeneity",
                     "return_claim_levels"):
            self.assertEqual(len((self.root / "analysis" / f"{name}.csv").read_text().splitlines()), 1, name)

    # ----- contrasts and derived metrics -----

    def test_generalized_contrasts_are_complete_case_per_contrast(self):
        conditions = ("event", "neutral", "no_calendar")
        records = (pair("s1", conditions=conditions, golds={"event": 50, "neutral": 40, "no_calendar": 30}) +
                   pair("s2", conditions=conditions, golds={"event": 20, "neutral": 10}))
        spoil(records[-1])  # s2 no_calendar is invalid
        report = self.run_analysis(records)
        self.assertEqual(report["contrasts"], ["event_minus_neutral", "event_minus_no_calendar",
                                               "neutral_minus_no_calendar"])
        primary = self.gold(report)
        self.assertEqual((primary["estimate"], primary["n_pairs"], primary["n_clusters"]), (10, 2, 2))
        control = self.gold(report, contrast="event_minus_no_calendar")
        self.assertEqual((control["estimate"], control["n_pairs"], control["n_clusters"]), (20, 1, 1))
        self.assertEqual(self.gold(report, contrast="neutral_minus_no_calendar")["estimate"], 10)
        self.assertFalse([r for r in report["effects"] if r["contrast"] == "event_minus_occasion_control"])
        coverage = {r["contrast"]: r for r in report["pair_coverage"]}
        self.assertEqual((coverage["event_minus_no_calendar"]["attempted_pairs"],
                          coverage["event_minus_no_calendar"]["complete_pairs"]), (2, 1))
        self.assertEqual(coverage["event_minus_neutral"]["complete_pairs"], 2)
        excluded = [r for r in report["exclusions"] if r["reason"] == "non_ok_pair"]
        self.assertEqual([r["contrast"] for r in excluded],
                         ["event_minus_no_calendar;neutral_minus_no_calendar"])

    def test_all_five_contrasts_and_signs(self):
        conditions = ("event", "neutral", "no_calendar", "occasion_control", "purchase_control")
        golds = {"event": 50, "neutral": 40, "no_calendar": 35, "occasion_control": 30, "purchase_control": 20}
        report = self.run_analysis(pair(conditions=conditions, golds=golds))
        expected = {"event_minus_neutral": 10, "event_minus_no_calendar": 15,
                    "event_minus_occasion_control": 20, "event_minus_purchase_control": 30,
                    "neutral_minus_no_calendar": 5}
        self.assertEqual(report["contrasts"], list(expected))
        for contrast, value in expected.items():
            self.assertEqual(self.gold(report, contrast=contrast)["estimate"], value)

    def test_derived_metrics_have_exact_values(self):
        # Reserve = 100000/200000 = 50% of capital; cash must cover it.
        event = payload(gold=40, equity=40, expected=6, wtp=12000)    # cash 20 -> shortfall 30
        neutral = payload(gold=30, equity=40, expected=4, wtp=5000)   # cash 30 -> shortfall 20
        report = self.run_analysis(pair(answers={"event": event, "neutral": neutral}))
        expected = {"return_claim_deviation_pp": 2, "reserve_shortfall_pct": 10,
                    "reserve_violation": 0, "wtp_premium_pct": 70}
        for metric, value in expected.items():
            self.assertAlmostEqual(self.row(report, metric)["estimate"], value, msg=metric)
        self.assertEqual(self.row(report, "return_claim_deviation_pp")["unit"], "percentage_points_annual_expected_return")
        self.assertEqual(self.row(report, "wtp_premium_pct")["unit"], "percent_of_gold_unit_price")
        # Event meets the reserve (cash 55 >= 50), neutral does not (cash 30).
        event = payload(gold=5, equity=40, expected=4, wtp=10000)
        report = self.run_analysis(pair(answers={"event": event, "neutral": neutral}))
        self.assertEqual(self.row(report, "reserve_violation")["estimate"], -1)
        self.assertAlmostEqual(self.row(report, "reserve_shortfall_pct")["estimate"], -20)
        self.assertAlmostEqual(self.row(report, "wtp_premium_pct")["estimate"], 50)

    def test_reserve_violation_threshold_is_one_hundredth_of_a_point(self):
        near = payload(gold=10, equity=40.005)  # cash 49.995 -> shortfall .005 (no violation)
        far = payload(gold=10, equity=40.02)    # cash 49.98 -> shortfall .02 (violation)
        report = self.run_analysis(pair(answers={"event": far, "neutral": near}))
        self.assertEqual(self.row(report, "reserve_violation")["estimate"], 1)

    def test_equal_wrong_return_claims_remain_visible_in_condition_levels(self):
        wrong = payload(expected=10)
        report = self.run_analysis(pair(answers={"event": wrong, "neutral": wrong}))
        self.assertEqual(self.row(report, "return_claim_deviation_pp")["estimate"], 0)
        levels = {(r["condition"], r["metric"]): r for r in report["return_claim_levels"]}
        for condition in ("event", "neutral"):
            self.assertEqual(levels[condition, "return_claim_deviation_pp"]["estimate"], 6)
            self.assertEqual(levels[condition, "return_claim_absolute_error_pp"]["estimate"], 6)
            self.assertEqual(levels[condition, "return_claim_violation"]["estimate"], 1)
        self.assertEqual(len(self.read_csv("return_claim_levels")), 6)
        signed = self.row(report, "return_claim_deviation_pp")
        self.assertEqual(signed["family"], "redundant_return_contrast")
        self.assertIsNone(signed["p_adjusted"])

    def test_return_error_contrast_distinguishes_correction_from_violation(self):
        report = self.run_analysis(pair(answers={"event": payload(expected=4), "neutral": payload(expected=10)}))
        self.assertEqual(self.row(report, "return_claim_absolute_error_pp")["estimate"], -6)
        self.assertEqual(self.row(report, "return_claim_violation")["estimate"], -1)

    def test_return_condition_levels_weight_scenarios_equally_and_exclude_rejected_records(self):
        records = (pair("s1", answers={"event": payload(expected=10)}) +
                   pair("s1", replicate=1, answers={"event": payload(expected=10)}) +
                   pair("s2", answers={"event": payload(expected=4)}))
        duplicate = pair("s3", answers={"event": payload(expected=90)})
        records += duplicate + [copy.deepcopy(duplicate[0])]
        report = self.run_analysis(records)
        event = next(r for r in report["return_claim_levels"]
                     if r["condition"] == "event" and r["metric"] == "return_claim_absolute_error_pp")
        self.assertEqual((event["estimate"], event["n_records"], event["n_clusters"]), (3, 3, 2))

    def test_return_tolerance_is_registered_validated_and_cannot_be_overridden(self):
        prereg = self.write_prereg({"schema_version": 1, "return_tolerance_pp": 0.5,
            "primary": [{"id": "return_error", "table": "effects", "metric": "return_claim_violation",
                         "contrast": "event_minus_neutral", "minimum_meaningful_effect": 0.1}]})
        records = pair(answers={"event": payload(expected=4.5), "neutral": payload(expected=3.49)})
        report = self.run_analysis(records, prereg_path=prereg)
        self.assertEqual(report["return_tolerance_pp"], 0.5)
        self.assertEqual(self.row(report, "return_claim_violation")["estimate"], -1)
        with self.assertRaisesRegex(ValueError, "preregistered tolerance"):
            self.run_analysis(records, prereg_path=prereg, return_tolerance_pp=0.1)
        for invalid in (-1, float("nan"), float("inf"), True, "0.1"):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "return_tolerance_pp"):
                self.run_analysis(records, return_tolerance_pp=invalid)
        exact = self.run_analysis(pair(answers={"event": payload(expected=4), "neutral": payload(expected=4.01)}))
        self.assertEqual(self.row(exact, "return_claim_violation")["estimate"], 0)

    def test_cli_passes_return_tolerance_to_analysis(self):
        from auspicious.cli import dispatch, parser
        path = self.root / "cli-records.jsonl"
        write_jsonl(path, pair(answers={"event": payload(expected=4.2), "neutral": payload(expected=4)}))
        args = parser().parse_args(["analyze", "--records", str(path), "--out", str(self.root / "cli-analysis"),
                                    "--bootstrap", "0", "--return-tolerance-pp", "0.25"])
        report = dispatch(args)
        self.assertEqual(report["return_tolerance_pp"], 0.25)
        self.assertEqual(self.row(report, "return_claim_violation")["estimate"], 0)

    # ----- interactions -----

    def test_persona_interaction_matches_tasks_and_excludes_reference(self):
        records = (pair("s1", 15, persona="observant") + pair("s1", 5, persona="nonobservant") +
                   pair("s2", 25, persona="observant") + pair("s2", 5, persona="nonobservant") +
                   pair("s3", 25, persona="observant"))
        report = self.run_analysis(records)
        row = self.row(report, "gold_pct", "persona_interactions")
        self.assertEqual(row["estimate"], 15)
        self.assertEqual((row["matched_pairs"], row["unmatched_tasks"], row["n_clusters"]), (2, 1, 2))
        self.assertEqual((row["dimension"], row["reference_level"], row["level"]), ("persona", "nonobservant", "observant"))
        self.assertEqual(row["contrast"], "event_minus_neutral")
        self.assertEqual({r["persona_id"] for r in report["persona_interactions"]}, {"observant"})

    def test_cue_interaction_against_festival_named(self):
        records = (pair("s1", 5, cue="festival_named") + pair("s1", 12, cue="explicit_belief") +
                   pair("s2", 15, cue="festival_named") + pair("s2", 20, cue="explicit_belief"))
        report = self.run_analysis(records)
        row = self.row(report, "gold_pct", "cue_interactions")
        self.assertEqual(row["estimate"], 6)
        self.assertEqual((row["dimension"], row["reference_level"], row["level"]), ("cue", "festival_named", "explicit_belief"))

    def test_interaction_skipped_when_reference_absent(self):
        report = self.run_analysis(pair(persona="observant") + pair("s2", persona="unspecified"))
        self.assertEqual(report["persona_interactions"], [])

    # ----- validity, bounds, robustness -----

    def test_validity_is_an_outcome_over_all_attempted_pairs(self):
        records = pair("s1") + pair("s3")
        for cluster in ("s2", "s5"):
            bad = pair(cluster)
            spoil(bad[0])
            records.extend(bad)
        bad = pair("s4")
        spoil(bad[1], "blocked")
        records.extend(bad)
        report = self.run_analysis(records)
        self.assertEqual(self.gold(report)["n_pairs"], 2)  # complete-case survivors only
        row = next(r for r in report["validity_effects"]
                   if r["row_type"] == "contrast_difference" and r["contrast"] == "event_minus_neutral")
        # per-pair P(ok event) - P(ok neutral): s1 0, s2 -1, s3 0, s4 +1, s5 -1
        self.assertAlmostEqual(row["estimate"], -.2)
        self.assertEqual((row["n_pairs"], row["n_clusters"]), (5, 5))
        rates = {(r["dimension"], r["level"]): r for r in report["validity_effects"] if r["row_type"] == "valid_rate"}
        event, neutral = rates[("condition", "event")], rates[("condition", "neutral")]
        self.assertEqual((event["n_records"], event["n_ok"], event["n_invalid"]), (5, 3, 2))
        self.assertAlmostEqual(event["valid_rate"], .6)
        self.assertEqual((neutral["n_ok"], neutral["n_blocked"]), (4, 1))
        self.assertEqual(rates[("all", "all")]["n_records"], 10)
        self.assertEqual(rates[("language", "en")]["n_records"], 10)
        self.assertTrue(any(r["row_type"] == "valid_rate" for r in self.read_csv("validity_effects")))

    def test_bounds_use_worst_case_over_attempted_pairs(self):
        bad = pair("s2")
        spoil(bad[0])
        records = pair("s1", 10) + bad + pair("s3", 20) + pair("s4")[:1]
        report = self.run_analysis(records)

        def bound(metric):
            return self.row(report, metric, "bounds_effects", contrast="event_minus_neutral")

        gold = bound("gold_pct")
        self.assertEqual((gold["n_pairs_attempted"], gold["n_pairs_complete"]), (4, 2))  # s4 has an absent neutral record
        self.assertAlmostEqual(gold["share_missing_one_side"], 1 / 2)
        self.assertAlmostEqual(gold["estimate_complete_case"], 15)
        # s2: event unknown in [0,100], neutral 30 -> pair difference in [-30, 70]
        self.assertAlmostEqual(gold["bound_low"], (10 - 30 + 20 - 60) / 4)
        self.assertAlmostEqual(gold["bound_high"], (10 + 70 + 20 + 40) / 4)
        self.assertFalse(gold["bounds_exclude_zero"])
        buy = bound("buy_now")
        self.assertAlmostEqual(buy["bound_low"], -1 / 4)
        self.assertAlmostEqual(buy["bound_high"], 1 / 4)
        timing = bound("timing_days")  # event unknown in [0, 30], neutral 7
        self.assertAlmostEqual(timing["bound_low"], -30 / 4)
        self.assertAlmostEqual(timing["bound_high"], 30 / 4)
        violation = bound("reserve_violation")  # event unknown in [0,1], neutral violates (1)
        self.assertAlmostEqual(violation["estimate_complete_case"], 0)
        self.assertAlmostEqual(violation["bound_low"], -1 / 4)
        self.assertAlmostEqual(violation["bound_high"], 1 / 4)
        skipped = bound("willingness_to_pay_inr")
        self.assertIsNone(skipped["bound_low"])
        self.assertEqual(skipped["skipped_reason"], "unbounded_metric")
        self.assertEqual(bound("return_claim_deviation_pp")["skipped_reason"], "unbounded_metric")

    def test_bounds_with_no_missingness_equal_the_estimate(self):
        report = self.run_analysis(pair("s1", 10) + pair("s2", 20))
        row = self.row(report, "gold_pct", "bounds_effects")
        self.assertEqual((row["bound_low"], row["bound_high"], row["estimate_complete_case"]), (15, 15, 15))
        self.assertTrue(row["bounds_exclude_zero"])
        self.assertEqual(row["share_missing_one_side"], 0)

    def test_scenario_robustness_and_design_summary(self):
        report = self.run_analysis(pair("s1", 10) + pair("s2", -5) + pair("s3", 20))
        row = self.row(report, "gold_pct", "scenario_robustness", contrast="event_minus_neutral")
        self.assertEqual(row["n_scenarios"], 3)
        self.assertAlmostEqual(row["pooled_estimate"], 25 / 3)
        self.assertAlmostEqual(row["share_same_sign"], 2 / 3)
        self.assertEqual((row["min_scenario_estimate"], row["max_scenario_estimate"]), (-5, 20))
        rows = [r for r in self.read_csv("scenario_effects") if r["metric"] == "gold_pct"]
        self.assertEqual(sorted(float(r["estimate"]) for r in rows), [-5, 10, 20])
        self.assertEqual(sorted(r["cluster_id"] for r in rows), ["s1", "s2", "s3"])
        summary = report["design_summary"]["model"]
        self.assertEqual((summary["n_scenarios"], summary["templates"], summary["replicates"]), (3, ["t1"], [0]))
        self.assertEqual((summary["distinct_prompt_texts"], summary["conditions"]), (6, ["event", "neutral"]))
        self.assertTrue(any("describe only the sampled scenarios" in w for w in report["warnings"]))

    def test_eight_scenarios_do_not_establish_adequate_inference(self):
        report = self.run_analysis([r for i in range(8) for r in pair(f"s{i}", i)])
        self.assertTrue(any("describe only the sampled scenarios" in w for w in report["warnings"]))

    def test_template_effects_only_when_several_templates(self):
        report = self.run_analysis(pair("s1", 10) + pair("s2", 10))
        self.assertEqual(len(self.read_csv("template_effects")), 0)
        records = (pair("s1", 10, template="t1") + pair("s1", 20, template="t2") +
                   pair("s2", 10, template="t1") + pair("s2", 20, template="t2"))
        report = self.run_analysis(records)
        self.assertEqual(self.gold(report)["estimate"], 15)  # pooled over templates within scenario
        rows = {r["template_id"]: r for r in self.read_csv("template_effects")
                if r["metric"] == "gold_pct" and r["contrast"] == "event_minus_neutral"}
        self.assertEqual({k: float(v["estimate"]) for k, v in rows.items()}, {"t1": 10, "t2": 20})
        self.assertEqual(report["design_summary"]["model"]["templates"], ["t1", "t2"])

    # ----- multiplicity and preregistration -----

    def test_holm_and_benjamini_hochberg_values(self):
        p = [0.01, 0.04, 0.03, 0.005]
        for got, want in zip(_holm(p), [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, want)
        for got, want in zip(_benjamini_hochberg(p), [0.02, 0.04, 0.04, 0.02]):
            self.assertAlmostEqual(got, want)
        self.assertEqual(_holm([0.9, 0.8]), [1.0, 1.0])  # 0.8*2 -> 1; 0.9*1 raised to the running max

    def test_descriptive_default_has_no_bootstrap_p_values(self):
        report = self.run_analysis(pair("s1", 10) + pair("s2", 10) + pair("s3", 10), bootstrap=100)
        gold = self.gold(report)
        self.assertIsNone(gold["p_value"])
        self.assertIsNone(self.row(report, "equity_pct")["p_value"])
        self.assertEqual((gold["family"], gold["adjustment"], gold["n_family"]), ("exploratory", "benjamini_hochberg", 11))
        # Unavailable tests stay in the family; no p-value is invented.
        self.assertIsNone(gold["p_adjusted"])
        cash = self.row(report, "cash_pct")
        self.assertTrue(cash["derived_from_allocation"])
        self.assertEqual(cash["family"], "derived_from_allocation")
        self.assertIsNone(cash["p_adjusted"])
        self.assertFalse(gold["derived_from_allocation"])
        self.assertIsNone(gold["exceeds_mme"])
        self.assertEqual(report["preregistration"]["provided"], False)
        self.assertEqual(report["preregistration"]["n_exploratory_tests"], 11)

    def test_retrospective_plan_cannot_create_p_values_or_mme_flags(self):
        prereg = self.write_prereg({
            "_note": "illustrative, not a preregistration", "schema_version": 1,
            "primary": [
                {"id": "h1", "table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
                 "filters": {"source": "model", "persona_id": "default"}, "minimum_meaningful_effect": 5},
                {"id": "h2", "table": "effects", "metric": "equity_pct", "contrast": "event_minus_neutral",
                 "filters": {"source": "model"}, "minimum_meaningful_effect": 5}],
            "primary_adjustment": "holm", "exploratory_adjustment": "holm"})
        report = self.run_analysis(pair("s1", 10) + pair("s2", 10) + pair("s3", 10), bootstrap=100,
                                   prereg_path=prereg)
        h1, h2 = self.gold(report), self.row(report, "equity_pct")
        self.assertEqual((h1["family"], h1["primary_id"], h1["adjustment"], h1["n_family"]), ("primary", "h1", "holm", 2))
        self.assertIsNone(h1["p_adjusted"])
        self.assertIsNone(h2["p_adjusted"])
        self.assertIsNone(h1["exceeds_mme"])
        self.assertIsNone(h1["within_mme"])
        self.assertEqual(h2["minimum_meaningful_effect"], 5)
        self.assertIsNone(h2["within_mme"])  # no equivalence claim without a valid interval
        self.assertIsNone(h2["exceeds_mme"])
        other = self.row(report, "buy_now")
        self.assertEqual((other["family"], other["adjustment"], other["n_family"]), ("exploratory", "holm", 9))
        summary = report["preregistration"]
        self.assertEqual(summary["sha256"], hashlib.sha256(prereg.read_bytes()).hexdigest())
        self.assertEqual((summary["n_primary_tests"], summary["n_exploratory_tests"]), (2, 9))
        self.assertEqual(summary["unmatched_primary"], [])

    def test_holm_keeps_untestable_rows_in_the_family(self):
        rows = [{"p_value": 0.03}, {"p_value": None}]
        self.assertEqual(_adjust_family(rows, "holm", "primary"), 2)
        self.assertEqual(rows[0]["p_adjusted"], 0.06)
        self.assertEqual([row["n_family"] for row in rows], [2, 2])
        self.assertIsNone(rows[1]["p_value"])
        self.assertIsNone(rows[1]["p_adjusted"])

    def test_primary_family_keeps_rows_without_enough_scenarios(self):
        prereg = self.write_prereg({"schema_version": 1, "primary": [{
            "id": "gold", "table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
            "minimum_meaningful_effect": 2, "expected_tests": 2}]})
        records = pair("s1") + pair("s2") + pair("s1", role="advisor")
        report = self.run_analysis(records, prereg_path=prereg)
        available = self.gold(report, role="investor")
        unavailable = self.gold(report, role="advisor")
        self.assertIsNone(available["p_adjusted"])
        self.assertEqual(unavailable["n_family"], 2)
        self.assertIsNone(unavailable["p_adjusted"])
        self.assertEqual(report["preregistration"]["n_unavailable_primary_tests"], 2)

    def test_expected_primary_tests_preserve_completely_absent_strata(self):
        document = {"schema_version": 1, "primary": [{
            "id": "gold", "table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
            "minimum_meaningful_effect": 2, "expected_tests": 2}]}
        report = self.run_analysis(pair("s1") + pair("s2"), prereg_path=self.write_prereg(document))
        self.assertEqual(self.gold(report)["n_family"], 2)
        self.assertIsNone(self.gold(report)["p_adjusted"])
        self.assertEqual(report["preregistration"]["entries"][0]["n_planned_tests"], 2)
        document["primary"][0]["expected_tests"] = 1
        with self.assertRaisesRegex(ValueError, "exceeding its expected_tests"):
            self.run_analysis(pair("s1") + pair("s1", role="advisor"),
                              prereg_path=self.write_prereg(document))

    def test_prereg_unmatched_primary_is_loud_but_outputs_remain(self):
        prereg = self.write_prereg({
            "schema_version": 1,
            "primary": [{"id": "hx", "table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
                         "filters": {"language": "hi"}, "minimum_meaningful_effect": 2}]})
        with self.assertRaisesRegex(ValueError, "matched no analysis row: hx"):
            self.run_analysis(pair("s1") + pair("s2"), prereg_path=prereg)
        written = json.loads((self.root / "analysis" / "analysis.json").read_text())
        self.assertEqual(written["preregistration"]["unmatched_primary"], ["hx"])
        self.assertTrue(any("PREREGISTRATION" in w for w in written["warnings"]))

    def test_prereg_schema_is_validated_strictly(self):
        good = {"id": "h1", "table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
                "filters": {}, "minimum_meaningful_effect": 2}
        bad_documents = [
            {"schema_version": 1, "primary": [good], "unknown": 1},
            {"schema_version": 2, "primary": [good]},
            {"schema_version": 1, "primary": []},
            {"schema_version": 1, "primary": [{**good, "table": "human_comparisons"}]},
            {"schema_version": 1, "primary": [{**good, "metric": "nonsense"}]},
            {"schema_version": 1, "primary": [{**good, "metric": "cash_pct"}]},
            {"schema_version": 1, "primary": [{**good, "metric": "return_claim_deviation_pp"}]},
            {"schema_version": 1, "primary": [{**good, "expected_tests": 0}]},
            {"schema_version": 1, "primary": [{**good, "expected_tests": True}]},
            {"schema_version": 1, "primary": [{**good, "expected_tests": None}]},
            {"schema_version": 1, "primary": [good], "return_tolerance_pp": -1},
            {"schema_version": 1, "primary": [{**good, "contrast": "event_minus_other"}]},
            {"schema_version": 1, "primary": [{**good, "filters": {"nonsense": "x"}}]},
            {"schema_version": 1, "primary": [{**good, "minimum_meaningful_effect": 0}]},
            {"schema_version": 1, "primary": [{**good, "_note": "only allowed at top level"}]},
            {"schema_version": 1, "primary": [good, good]},
            {"schema_version": 1, "primary": [good], "exploratory_adjustment": "bonferroni"},
            {"schema_version": 1, "primary": [good], "primary_adjustment": "bh"},
        ]
        for document in bad_documents:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    self.run_analysis(pair(), prereg_path=self.write_prereg(document))

    def test_prereg_overlapping_entries_are_rejected(self):
        entry = {"table": "effects", "metric": "gold_pct", "contrast": "event_minus_neutral",
                 "filters": {}, "minimum_meaningful_effect": 2}
        prereg = self.write_prereg({"schema_version": 1, "primary": [{**entry, "id": "a"}, {**entry, "id": "b"}]})
        with self.assertRaisesRegex(ValueError, "several primary entries"):
            self.run_analysis(pair("s1") + pair("s2"), prereg_path=prereg)

    def test_shipped_prereg_example_is_valid_schema(self):
        from auspicious.analysis import _load_prereg
        example = Path(__file__).resolve().parent.parent / "examples" / "prereg-example.json"
        loaded = _load_prereg(example)
        self.assertTrue(loaded["primary"])
        self.assertIn("not externally registered", json.loads(example.read_text())["_note"])

    # ----- human comparison -----

    def test_human_within_comparison_matches_tasks_and_bootstraps_subjects_scenarios(self):
        models = pair("s1", 15) + pair("s2", 15)
        humans = []
        for subject in ("a", "b"):
            humans.extend(pair("s1", 5, subject=subject) + pair("s2", 5, subject=subject))
        report = self.run_analysis(models, humans)
        self.assertEqual(report["human_design"], "within")
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertEqual(row["estimate"], 10)
        self.assertEqual((row["n_subjects"], row["n_clusters"], row["matched_human_units"]), (2, 2, 4))
        self.assertEqual((row["ci_low"], row["bootstrap_valid_draws"], row["human_design"]), (None, 0, "within"))
        human_row = self.row(report, "gold_pct", source="human")
        self.assertEqual((human_row["estimate"], human_row["n_subjects"]), (5, 2))

    def test_human_comparison_rejects_same_ids_with_different_role(self):
        models, humans = pair(), pair(subject="a")
        for record in humans:
            record["trial"]["role"] = "advisor"
        report = self.run_analysis(models, humans)
        self.assertTrue(all(r["matched_human_units"] == 0 for r in report["human_comparisons"]))
        self.assertIn("human_task_metadata_mismatch", [r["reason"] for r in report["exclusions"]])

    def test_within_requires_subject_to_supply_both_conditions(self):
        humans = [pair(subject="a")[0], pair(subject="b")[1]]
        report = self.run_analysis(pair(), humans, human_design="within")
        self.assertTrue(all(r["matched_human_units"] == 0 for r in report["human_comparisons"]))

    def test_duplicate_human_observations_are_rejected(self):
        humans = pair(subject="a")
        humans.append(copy.deepcopy(humans[0]))
        with self.assertRaisesRegex(ValueError, "Duplicate human"):
            self.run_analysis(pair(), humans)

    def unpaired_fixture(self):
        s1, s2 = pair("s1", 15), pair("s2", 15)
        humans = [human(s1, "event", "a", 50), human(s1, "event", "b", 40),
                  human(s1, "neutral", "c", 30), human(s1, "neutral", "d", 20),
                  human(s2, "event", "a", 35), human(s2, "event", "b", 25),
                  human(s2, "neutral", "c", 20), human(s2, "neutral", "d", 10)]
        return s1 + s2, humans

    def test_unpaired_estimator_hand_computed(self):
        models, humans = self.unpaired_fixture()
        report = self.run_analysis(models, humans, bootstrap=200)
        self.assertEqual(report["human_design"], "unpaired")  # auto: every subject saw single-condition pairs
        # s1: event mean (50+40)/2 = 45, neutral mean (30+20)/2 = 25 -> 20
        # s2: event mean (35+25)/2 = 30, neutral mean (20+10)/2 = 15 -> 15
        effect = self.row(report, "gold_pct", source="human")
        self.assertEqual((effect["estimate"], effect["n_pairs"], effect["n_clusters"], effect["n_subjects"]), (17.5, 2, 2, 4))
        # model effect 15 in both scenarios: ((15-20) + (15-15)) / 2
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertEqual(row["estimate"], -2.5)
        self.assertEqual((row["human_design"], row["matched_model_pairs"], row["matched_human_units"]), ("unpaired", 2, 2))
        self.assertEqual((row["n_clusters"], row["n_subjects"], row["bootstrap_valid_draws"]), (2, 4, 0))
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["ci_method"], "unavailable")
        self.assertEqual(next(r for r in report["pair_coverage"] if r["source"] == "human")["complete_pairs"], 2)

    def test_unpaired_single_subject_per_arm_has_no_population_interval(self):
        models, humans = [], []
        for i in range(12):
            records = pair(f"s{i}", 10)
            models.extend(records)
            humans.extend([human(records, "event", "event_subject", 40),
                           human(records, "neutral", "neutral_subject", 30)])
        report = self.run_analysis(models, humans, bootstrap=200)
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertEqual((row["estimate"], row["n_subjects"], row["n_clusters"]), (0, 2, 12))
        self.assertIsNone(row["ci_low"])
        self.assertIsNone(row["ci_high"])
        self.assertEqual(row["bootstrap_valid_draws"], 0)
        self.assertEqual(row["ci_unavailable_reason"], "descriptive_observed_participants_only")

    def test_unpaired_total_subject_count_does_not_hide_a_singleton_arm(self):
        models, humans = self.unpaired_fixture()
        humans = [r for r in humans if r["result"]["subject_id"] != "b"]
        report = self.run_analysis(models, humans)
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertEqual(row["n_subjects"], 3)
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["ci_unavailable_reason"], "descriptive_observed_participants_only")

    def test_counterbalanced_singleton_stratum_has_no_population_interval(self):
        s1, s2 = pair("s1"), pair("s2")
        humans = [human(s1, "event", "a", 40), human(s2, "neutral", "a", 30),
                  human(s1, "event", "b", 45), human(s2, "neutral", "b", 35),
                  human(s1, "neutral", "c", 20), human(s2, "event", "c", 50)]
        report = self.run_analysis(s1 + s2, humans)
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertEqual(row["n_subjects"], 3)
        self.assertIsNone(row["ci_low"])  # observed-participant estimates remain descriptive
        humans.append(human(s1, "neutral", "d", 25))  # a singleton neutral-only stratum
        report = self.run_analysis(s1 + s2, humans)
        row = next(r for r in report["human_comparisons"] if r["metric"] == "gold_pct")
        self.assertIsNone(row["ci_low"])
        self.assertEqual(row["ci_unavailable_reason"], "descriptive_observed_participants_only")

    def test_unpaired_requires_valid_response_on_each_side(self):
        models, humans = self.unpaired_fixture()
        humans = [h for h in humans if not (h["trial"]["scenario_id"] == "s2" and h["trial"]["condition"] == "neutral")]
        humans.append(spoil(human(pair("s2", 15), "neutral", "c", 20)))
        report = self.run_analysis(models, humans)
        effect = self.row(report, "gold_pct", source="human")
        self.assertEqual((effect["estimate"], effect["n_pairs"]), (20, 1))
        self.assertIn("non_ok_pair", [r["reason"] for r in report["exclusions"] if r["source"] == "human"])

    def test_unpaired_ignores_invalid_responses_without_imputing(self):
        models, humans = self.unpaired_fixture()
        humans.append(spoil(human(pair("s1", 15), "event", "e", 0)))
        report = self.run_analysis(models, humans)
        self.assertEqual(self.row(report, "gold_pct", source="human")["estimate"], 17.5)

    def test_auto_design_detects_within_and_rejects_mixed(self):
        mixed = pair("s1", subject="a") + [human(pair("s2"), "event", "b", 40)]
        with self.assertRaisesRegex(ValueError, "cannot decide"):
            self.run_analysis(pair("s1") + pair("s2"), mixed)
        report = self.run_analysis(pair("s1") + pair("s2"), mixed, human_design="within")
        self.assertEqual(report["human_design"], "within")

    def test_explicit_unpaired_rejects_subject_with_both_conditions(self):
        with self.assertRaisesRegex(ValueError, "answered both conditions"):
            self.run_analysis(pair(), pair(subject="a"), human_design="unpaired")

    def test_invalid_design_name(self):
        with self.assertRaisesRegex(ValueError, "human_design"):
            self.run_analysis(pair(), human_design="paired")

    # ----- participants -----

    def participant_rows(self):
        return [
            {"subject_id": "a", "survey_language": "en", "language_proficiency": "5", "financial_literacy": "2.0",
             "observance_self_report": "5", "age_band": "18-24", "region": "north"},
            {"subject_id": "b", "survey_language": "en", "language_proficiency": "4", "financial_literacy": "8",
             "observance_self_report": "4", "age_band": "", "region": ""},
            {"subject_id": "c", "survey_language": "en", "language_proficiency": "3", "financial_literacy": "3",
             "observance_self_report": "1", "age_band": "", "region": ""},
            {"subject_id": "d", "survey_language": "hi", "language_proficiency": "5", "financial_literacy": "9",
             "observance_self_report": "3", "age_band": "", "region": ""},
            {"subject_id": "e", "survey_language": "hi", "language_proficiency": "2", "financial_literacy": "1",
             "observance_self_report": "2", "age_band": "", "region": ""},
        ]

    def test_import_participants_validates_strictly(self):
        path = self.csv_file("participants.csv", PARTICIPANT_FIELDS, self.participant_rows())
        output = self.root / "participants.json"
        summary = import_participants(path, output)
        self.assertEqual((summary["participants"], summary["by_survey_language"]), (5, {"en": 3, "hi": 2}))
        document = json.loads(output.read_text())
        self.assertEqual(document["participants"][0]["age_band"], "18-24")
        self.assertIsNone(document["participants"][1]["age_band"])
        self.assertEqual(document["participants"][1]["financial_literacy"], 8)
        # header-only and optional columns omitted are valid
        empty = self.csv_file("empty.csv", PARTICIPANT_FIELDS[:5], [])
        self.assertEqual(import_participants(empty, self.root / "empty.json")["participants"], 0)
        rows = [{k: v for k, v in row.items() if k in PARTICIPANT_FIELDS[:5]} for row in self.participant_rows()[:1]]
        self.assertEqual(import_participants(self.csv_file("short.csv", PARTICIPANT_FIELDS[:5], rows),
                                             self.root / "short.json")["participants"], 1)
        first = self.participant_rows()[0]
        for label, rows, fields in (
            ("duplicate", [first, first], PARTICIPANT_FIELDS),
            ("language", [{**first, "survey_language": "fr"}], PARTICIPANT_FIELDS),
            ("proficiency range", [{**first, "language_proficiency": "6"}], PARTICIPANT_FIELDS),
            ("observance float", [{**first, "observance_self_report": "4.5"}], PARTICIPANT_FIELDS),
            ("blank literacy", [{**first, "financial_literacy": ""}], PARTICIPANT_FIELDS),
            ("nonfinite literacy", [{**first, "financial_literacy": "nan"}], PARTICIPANT_FIELDS),
            ("blank subject", [{**first, "subject_id": " "}], PARTICIPANT_FIELDS),
            ("missing column", [{k: v for k, v in first.items() if k != "observance_self_report"}],
             tuple(f for f in PARTICIPANT_FIELDS if f != "observance_self_report")),
            ("unknown column", [{**first, "income": "1"}], (*PARTICIPANT_FIELDS, "income")),
        ):
            with self.subTest(label):
                with self.assertRaises(ValueError):
                    import_participants(self.csv_file("bad.csv", fields, rows), self.root / "bad.json")

    def test_participants_require_human_data(self):
        path = self.csv_file("participants.csv", PARTICIPANT_FIELDS, self.participant_rows())
        import_participants(path, self.root / "participants.json")
        with self.assertRaisesRegex(ValueError, "requires human_path"):
            self.run_analysis(pair(), participants_path=self.root / "participants.json")

    def test_participant_tables_and_heterogeneity_splits(self):
        path = self.csv_file("participants.csv", PARTICIPANT_FIELDS, self.participant_rows())
        participants = self.root / "participants.json"
        import_participants(path, participants)
        humans = []
        # within design: each subject supplies both conditions of each pair
        for subject, difference in (("a", 10), ("b", 30), ("c", 0), ("d", 5), ("z", 20)):
            humans.extend(pair("s1", difference, subject=subject) + pair("s2", difference, subject=subject))
        report = self.run_analysis(pair("s1", 15) + pair("s2", 15), humans, participants_path=participants)
        self.assertEqual(report["human_design"], "within")

        def split(covariate, level, language="all", metric="gold_pct"):
            return next(r for r in report["human_heterogeneity"]
                        if (r["covariate"], r["level"], r["survey_language"], r["metric"]) == (covariate, level, language, metric))

        # observance high (>=4) {a:10, b:30} -> 20; low (<=2) {c:0} -> 0; d (3) is middle and excluded
        self.assertEqual(split("observance_self_report", "high")["estimate"], 20)
        self.assertEqual(split("observance_self_report", "low")["estimate"], 0)
        self.assertEqual(split("observance_self_report", "high")["n_middle_excluded"], 1)
        self.assertEqual(split("observance_self_report", "high")["n_subjects_contributing"], 2)
        # literacy of subjects with responses a2 b8 c3 d9: median 5.5 -> high {b, d} -> 17.5; low {a, c} -> 5
        self.assertEqual(split("financial_literacy", "high")["estimate"], 17.5)
        self.assertEqual(split("financial_literacy", "high")["split_value"], 5.5)
        self.assertEqual(split("financial_literacy", "low")["estimate"], 5)
        self.assertEqual(split("financial_literacy", "high", "hi")["estimate"], 5)  # only d is a Hindi-survey subject
        self.assertEqual(split("observance_self_report", "high")["human_design"], "within")
        missing = [r for r in report["exclusions"] if r["reason"] == "subject_missing_from_participants_file"]
        self.assertEqual([r["detail"] for r in missing], ["z"])
        table = {r["survey_language"]: r for r in report["human_participants"]}
        self.assertEqual((table["all"]["n_participants"], table["all"]["n_with_responses"]), (5, 4))
        self.assertEqual((table["all"]["observance_high_n"], table["all"]["observance_middle_n"],
                          table["all"]["observance_low_n"]), (2, 1, 2))
        self.assertEqual((table["en"]["n_participants"], table["hi"]["n_with_responses"]), (3, 1))
        self.assertAlmostEqual(table["all"]["financial_literacy_median"], 3)
        self.assertTrue(self.read_csv("human_participants"))
        self.assertTrue(self.read_csv("human_heterogeneity"))

    def test_heterogeneity_uses_the_unpaired_estimator_for_unpaired_data(self):
        models, humans = self.unpaired_fixture()
        rows = [{"subject_id": s, "survey_language": "en", "language_proficiency": "5", "financial_literacy": str(lit),
                 "observance_self_report": str(obs), "age_band": "", "region": ""}
                for s, lit, obs in (("a", 1, 5), ("b", 2, 5), ("c", 3, 1), ("d", 4, 1))]
        participants = self.root / "participants.json"
        import_participants(self.csv_file("participants.csv", PARTICIPANT_FIELDS, rows), participants)
        report = self.run_analysis(models, humans, participants_path=participants)
        row = next(r for r in report["human_heterogeneity"] if (r["covariate"], r["level"], r["survey_language"], r["metric"])
                   == ("observance_self_report", "high", "all", "gold_pct"))
        # high-observance subjects a, b only answered event trials, so no pair has both sides
        self.assertIsNone(row["estimate"])
        self.assertEqual(row["human_design"], "unpaired")
        self.assertEqual(row["n_pairs"], 0)

    # ----- imports -----

    def test_import_humans_retains_invalid_rows_without_imputation(self):
        trials = [r["trial"] for r in pair()]
        trial_path = self.root / "trials.jsonl"
        write_jsonl(trial_path, trials)
        rows = [{"trial_id": trial["trial_id"], "subject_id": "anon-1", **payload()} for trial in trials]
        rows[1]["gold_pct"] = "refused"
        csv_path = self.csv_file("human.csv", HUMAN_FIELDS, rows)
        output = self.root / "imported.jsonl"
        summary = legacy_import_humans(trial_path, csv_path, output)
        self.assertEqual(summary["status_counts"]["ok"], 1)
        self.assertEqual(summary["status_counts"]["invalid"], 1)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertIsNone(records[1]["result"]["payload"])
        self.assertEqual(records[1]["result"]["source_cells"]["gold_pct"], "refused")

    def test_import_humans_header_only_and_duplicate_and_unknown_ids(self):
        trials = [r["trial"] for r in pair()]
        trial_path = self.root / "trials.jsonl"
        write_jsonl(trial_path, trials)
        csv_path = self.csv_file("human.csv", HUMAN_FIELDS, [])
        output = self.root / "imported.jsonl"
        self.assertEqual(legacy_import_humans(trial_path, csv_path, output)["records"], 0)
        row = {"trial_id": trials[0]["trial_id"], "subject_id": "anon-1", **payload()}
        self.csv_file("human.csv", HUMAN_FIELDS, [row, row])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            legacy_import_humans(trial_path, csv_path, output)
        self.csv_file("human.csv", HUMAN_FIELDS, [{**row, "trial_id": "unknown"}])
        with self.assertRaisesRegex(ValueError, "unknown trial_id"):
            legacy_import_humans(trial_path, csv_path, output)

    def test_import_human_schema_violations_become_invalid(self):
        trials = [r["trial"] for r in pair()]
        trial_path = self.root / "trials.jsonl"
        write_jsonl(trial_path, trials)
        base = {"trial_id": trials[0]["trial_id"], "subject_id": "a", **payload()}
        rows = [{**base, "subject_id": "a", "gold_pct": "NaN"},
                {**base, "subject_id": "b", "buy_now": "yes"},
                {**base, "subject_id": "c", "timing_days": "1.5"},
                {**base, "subject_id": "d", "cash_pct": "90"}]
        path = self.csv_file("human.csv", HUMAN_FIELDS, rows)
        result = legacy_import_humans(trial_path, path, self.root / "imported.jsonl")
        self.assertEqual(result["status_counts"]["invalid"], 4)

    def test_shipped_csv_headers(self):
        examples = Path(__file__).resolve().parent.parent / "examples"
        self.assertEqual(tuple((examples / "human-format.csv").read_text().strip().split(",")),
                         ("study_id", "subject_id", "trial_id", "position", "submitted_at", *HUMAN_FIELDS[2:]))
        self.assertEqual(tuple((examples / "participants-format.csv").read_text().strip().split(",")), PARTICIPANT_FIELDS)

    # ----- contextual series -----

    def test_context_preserves_units_sources_and_annotations(self):
        rows = [{"date": "2020-01-01", "series": "fixture-gold", "value": 100,
                 "unit": "INR/gram", "source": "test-source-a", "event": "hypothetical"},
                {"date": "2020-01-02", "series": "fixture-gold", "value": 110,
                 "unit": "INR/gram", "source": "test-source-a", "event": ""},
                {"date": "2020-01-01", "series": "fixture-gold", "value": 5,
                 "unit": "tonnes", "source": "test-source-a", "event": ""},
                {"date": "2020-01-01", "series": "fixture-gold", "value": 99,
                 "unit": "INR/gram", "source": "test-source-b", "event": ""}]
        path = self.csv_file("context.csv", CONTEXT_FIELDS, rows)
        report = contextual_series(path, self.root / "context")
        self.assertEqual(len(report["series"]), 3)
        selected = next(r for r in report["series"] if r["n_observations"] == 2)
        self.assertEqual(selected["absolute_change"], 10)
        self.assertEqual(selected["unit"], "INR/gram")
        self.assertIn("not individual", report["interpretation"])
        self.assertIn("hypothetical", (self.root / "context" / "contextual_data.csv").read_text())

    def test_context_header_only_is_valid(self):
        path = self.csv_file("context.csv", CONTEXT_FIELDS, [])
        report = contextual_series(path, self.root / "context")
        self.assertEqual(report["observations"], 0)
        self.assertEqual(report["series"], [])
        self.assertEqual(len((self.root / "context" / "contextual_data.csv").read_text().splitlines()), 1)

    def test_context_rejects_nonfinite_duplicate_and_non_iso_date(self):
        row = {"date": "2020-01-01", "series": "fixture", "value": 5,
               "unit": "INR", "source": "test", "event": ""}
        for rows in ([{**row, "value": "NaN"}], [row, row], [{**row, "date": "20200101"}]):
            with self.subTest(rows=rows):
                path = self.csv_file("context.csv", CONTEXT_FIELDS, rows)
                with self.assertRaises(ValueError):
                    contextual_series(path, self.root / "context")

    def test_csv_missing_headers_is_error(self):
        path = self.csv_file("context.csv", ("date", "value"), [])
        with self.assertRaisesRegex(ValueError, "missing CSV columns"):
            contextual_series(path, self.root / "context")


if __name__ == "__main__":
    unittest.main()
