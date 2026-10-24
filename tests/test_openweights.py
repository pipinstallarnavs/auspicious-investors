"""Offline invariants; optional runtime tests use a real, tiny random Llama.

Synthetic fixtures below test numerical behavior only. They are not observations,
validated calendar stimuli, or evidence about any trained model's behavior.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from auspicious import openweights as ow

try:
    import numpy as np
except ImportError:
    np = None


def provenance(width=3):
    return {
        "format_version": 1, "model": "local-fixture", "revision": "fixture-revision",
        "resolved_revision": "fixture-commit", "layer": 1, "width": width,
        "local_model_sha256": None,
        "representation": "decoder_block_output_prompt_final_token",
        "prompt_format": "tokenizer_chat_template_if_available_else_plain",
        "trust_remote_code": False,
    }


def activation_fixture(path, prompts="separable", template_by_cluster=False):
    """Train/dev/test clusters of four rows (labels 0,1,0,1); trials stored beside the features."""
    rows, labels, clusters, splits, trials = [], [], [], [], []
    for split, count in (("train", 3), ("dev", 2), ("test", 2)):
        for cluster in range(count):
            for k, label in enumerate((0, 1, 0, 1)):
                nuisance = cluster if split == "train" else 10000 + cluster
                rows.append([2 * (2 * label - 1), nuisance, 7])
                labels.append(label)
                clusters.append(f"{split}-{cluster}")
                splits.append(split)
                if prompts == "separable":
                    base = "Dhanteras gold festival offer today" if label else "ordinary working day schedule"
                    text = f"{base} scenario {split}{cluster} part {k // 2}"
                else:
                    # Identical text in both classes: nothing in the prompt reveals the label.
                    text = f"scenario {split}{cluster} part {k // 2}"
                trials.append({
                    "trial_id": f"trial-{len(rows) - 1}", "cluster_id": f"{split}-{cluster}", "split": split,
                    "condition": "event" if label else "neutral", "prompt": text,
                    "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    "festival": "dhanteras" if cluster % 2 == 0 else "muhurat",
                    "language": "en" if k < 2 else "hi",
                    "template_id": f"t{cluster % 2 + 1}" if template_by_cluster else "t1",
                    "paraphrase_id": "direct", "event_valence": "auspicious",
                })
    arrays = {
        "features": np.asarray(rows, dtype=float), "labels": np.asarray(labels),
        "cluster_ids": np.asarray(clusters), "splits": np.asarray(splits),
        "trial_ids": np.asarray([f"trial-{i}" for i in range(len(rows))]),
        "trials": np.asarray(json.dumps(trials)),
        "metadata": np.asarray(json.dumps(dict(provenance(), kind="prompt_activations", event_valences=["auspicious"], trials_sha256=ow._trials_digest(trials)))),
    }
    np.savez(path, **arrays)
    return arrays


def v2_trial(condition, prompt, pair="fixture-pair", cluster="fixture-scenario", split="train"):
    return {
        "trial_id": hashlib.sha256(f"{condition}:{prompt}".encode()).hexdigest(), "pair_id": pair,
        "cluster_id": cluster, "scenario_id": cluster, "split": split,
        "language": "en", "festival": "dhanteras", "condition": condition,
        "event_valence": "auspicious", "cue": "festival_named", "role": "investor",
        "persona_id": "fixture-persona", "paraphrase_id": "direct", "template_id": "t1", "replicate": 0,
        "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "review_status": "draft", "review": {},
    }


class ProviderConfigurationTests(unittest.TestCase):
    def test_import_and_constructor_are_lazy(self):
        script = (
            "import sys; from auspicious.openweights import OpenWeightProvider; "
            "p=OpenWeightProvider('explicit-model'); "
            "assert all(x not in sys.modules for x in ('torch','transformers','numpy')); "
            "assert p.identity['requested_model']=='explicit-model'"
        )
        completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_configuration_rejects_unbacked_interventions(self):
        for kwargs in ({"strength": 1}, {"random_direction_seed": 0},
                       {"temperature": float("nan")}, {"max_output_tokens": 0},
                       {"seed": -1}, {"steering_positions": "everywhere"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ow.OpenWeightProvider("explicit-model", **kwargs)

    def test_steering_positions_are_part_of_identity(self):
        identities = {}
        for position in ("all", "prompt_last", "generated"):
            provider = ow.OpenWeightProvider("explicit-model", steering_positions=position)
            self.assertEqual(provider.identity["generation"]["steering_positions"], position)
            identities[position] = json.dumps(provider.identity, sort_keys=True)
        self.assertEqual(len(set(identities.values())), 3)
        self.assertEqual(ow.OpenWeightProvider("explicit-model").identity["generation"]["steering_positions"], "all")

    def test_unsupported_architecture_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Unsupported architecture"):
            ow._decoder_layers(object())

    def test_layer_bounds_and_negative_resolution(self):
        self.assertEqual(ow._layer_index(-1, 3), 2)
        self.assertEqual(ow._layer_index(-3, 3), 0)
        for index in (3, -4, True, 1.5):
            with self.assertRaises(ValueError):
                ow._layer_index(index, 3)

    def test_local_model_and_tokenizer_bytes_bind_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "model.safetensors").write_bytes(b"synthetic-weight-fixture")
            tokenizer = root / "tokenizer.json"
            tokenizer.write_text('{"fixture":1}', encoding="utf-8")
            first = ow.OpenWeightProvider(folder).identity
            self.assertEqual(first, ow.OpenWeightProvider(folder).identity)
            tokenizer.write_text('{"fixture":2}', encoding="utf-8")
            second = ow.OpenWeightProvider(folder).identity
            self.assertNotEqual(first["local_model_sha256"], second["local_model_sha256"])
            (root / "model.safetensors").write_bytes(b"changed-weight-fixture")
            third = ow.OpenWeightProvider(folder).identity
            self.assertNotEqual(second["local_model_sha256"], third["local_model_sha256"])


class CapabilityDataTests(unittest.TestCase):
    @staticmethod
    def independent_answer(item):
        p = item["params"]
        if item["kind"] == "percent_of":
            return p["amount"] * p["percent"] / 100
        if item["kind"] == "compound_growth":
            value = p["principal"]
            for _ in range(p["years"]):
                value = value * (1 + p["rate_pct"] / 100)
            return value
        if item["kind"] == "cash_remainder":
            return p["total"] - p["total"] * p["gold_pct"] / 100 - p["total"] * p["equity_pct"] / 100
        if item["kind"] == "return_difference":
            return p["principal"] * p["rate_a_pct"] / 100 - p["principal"] * p["rate_b_pct"] / 100
        if item["kind"] == "expected_value":
            return p["gain"] * p["prob_pct"] / 100 - p["loss"] * (100 - p["prob_pct"]) / 100
        raise AssertionError(item["kind"])

    def setUp(self):
        self.data, self.sha, self.path = ow._load_capability(None)

    def test_default_path_is_package_data_and_draft(self):
        self.assertEqual(Path(self.path), Path(ow.__file__).resolve().parent.parent / "data" / "capability.json")
        self.assertTrue(self.data["status"].startswith("DRAFT"))
        self.assertEqual(self.sha, hashlib.sha256(Path(self.path).read_bytes()).hexdigest())

    def test_coverage_and_recomputed_answers(self):
        self.assertEqual(list(self.data["languages"]), ["en", "hi", "ta", "gu", "bn"])
        self.assertGreaterEqual(len(self.data["items"]), 12)
        self.assertEqual(len({item["id"] for item in self.data["items"]}), len(self.data["items"]))
        self.assertEqual({item["kind"] for item in self.data["items"]},
                         {"percent_of", "compound_growth", "cash_remainder", "return_difference", "expected_value"})
        for item in self.data["items"]:
            self.assertAlmostEqual(item["answer"], self.independent_answer(item), places=6, msg=item["id"])

    def test_prompts_render_in_every_language_with_latin_digits(self):
        for language in self.data["languages"]:
            for item in self.data["items"]:
                prompt = ow._capability_prompt(self.data, language, item)
                self.assertIn('{"answer"', prompt)
                self.assertNotIn("{amount", prompt)
                for char in prompt:
                    self.assertTrue(not char.isdigit() or char in "0123456789", (language, item["id"], char))
                for value in item["params"].values():
                    self.assertIn(str(value), prompt, (language, item["id"]))

    def test_tampered_answer_or_missing_draft_marker_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "items.json"
            tampered = json.loads(json.dumps(self.data))
            tampered["items"][0]["answer"] += 1
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "recomputed"):
                ow._load_capability(path)
            approved = json.loads(json.dumps(self.data))
            approved["status"] = "approved"
            path.write_text(json.dumps(approved), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "DRAFT"):
                ow._load_capability(path)

    def test_strict_answer_parsing_and_tolerance(self):
        self.assertEqual(ow._parse_answer(' {"answer": 12.5} ')[0], 12.5)
        for raw in ('```json\n{"answer": 1}\n```', "[1]", '{"answer": true}', '{"answer": "3"}',
                    '{"answer": 1, "extra": 2}', '{"answer": NaN}', '{"answer": 1e999}', "", "{}"):
            with self.subTest(raw=raw):
                self.assertIsNone(ow._parse_answer(raw)[0])
        self.assertTrue(ow._answer_correct(100.05, 100))
        self.assertFalse(ow._answer_correct(100.2, 100))
        self.assertTrue(ow._answer_correct(0.005, 0))
        self.assertFalse(ow._answer_correct(0.02, 0))

    def test_capability_cache_binds_content_path_limit_and_provider(self):
        class Fake:
            identity = {"provider": "fake", "generation": {"strength": 0.0}}

            def generate_text(self, prompt, key=None):
                return {"status": "ok", "raw_text": '{"answer": 0}'}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            items, target = root / "items.json", root / "report.json"
            items.write_text(json.dumps(self.data), encoding="utf-8")
            provider = Fake()
            with patch.object(ow, "capability_check", wraps=ow.capability_check) as run:
                original = ow._capability_report(target, provider, items_path=items, limit=1)
                before = target.read_bytes()
                ow._capability_report(target, provider, items_path=items, limit=1)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(target.read_bytes(), before)
                self.data["languages"]["en"]["instruction"] += " Updated question wording."
                items.write_text(json.dumps(self.data), encoding="utf-8")
                updated = ow._capability_report(target, provider, items_path=items, limit=1)
                self.assertEqual(run.call_count, 2)
                self.assertNotEqual(updated["items_sha256"], original["items_sha256"])
                alternate = root / "alternate.json"
                alternate.write_bytes(items.read_bytes())
                moved = ow._capability_report(target, provider, items_path=alternate, limit=1)
                self.assertEqual(run.call_count, 3)
                self.assertEqual(moved["items_path"], str(alternate))
                ow._capability_report(target, provider, items_path=alternate, limit=2)
                self.assertEqual(run.call_count, 4)
                provider.identity = {"provider": "fake", "generation": {"strength": 1.0}}
                ow._capability_report(target, provider, items_path=alternate, limit=2)
                self.assertEqual(run.call_count, 5)
                del self.data["status"]
                alternate.write_text(json.dumps(self.data), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "DRAFT"):
                    ow._capability_report(target, provider, items_path=alternate, limit=2)
                self.assertEqual(run.call_count, 5)

    def test_capability_report_with_known_replies(self):
        gold = {(language, ow._capability_prompt(self.data, language, item)): item["answer"]
                for language in self.data["languages"] for item in self.data["items"]}

        class Fake:
            identity = {"provider": "fake", "generation": {"strength": 0.0, "steering_positions": "all"}}

            def generate_text(inner, prompt, key=None):
                language = key.split(":")[1]
                answer = gold[(language, prompt)]
                raw = {"en": json.dumps({"answer": answer}), "hi": json.dumps({"answer": answer * 2 + 1}),
                       "ta": "not json", "gu": json.dumps({"answer": answer * 1.0005}),
                       "bn": '{"answer": 1, "extra": 2}'}[language]
                return {"status": "ok", "raw_text": raw, "usage": {"output_tokens": 1}}

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "capability.json"
            report = ow.capability_check(target, Fake())
            count = len(self.data["items"])
            self.assertEqual(report["overall"]["n"], 5 * count)
            expected = {"en": (1.0, 0.0), "hi": (0.0, 0.0), "ta": (0.0, 1.0), "gu": (1.0, 0.0), "bn": (0.0, 1.0)}
            for language, (accuracy, invalid) in expected.items():
                self.assertEqual(report["per_language"][language]["accuracy"], accuracy, language)
                self.assertEqual(report["per_language"][language]["invalid_rate"], invalid, language)
            self.assertAlmostEqual(report["overall"]["accuracy"], 0.4)
            self.assertAlmostEqual(report["overall"]["invalid_rate"], 0.4)
            self.assertEqual(report["provider_identity"], Fake.identity)
            self.assertTrue(report["items_status"].startswith("DRAFT"))
            saved = json.loads(target.read_text(encoding="utf-8"))
            self.assertEqual(saved["overall"], report["overall"])
            limited = ow.capability_check(target, Fake(), limit=4)
            self.assertEqual(limited["overall"]["n"], 20)
            self.assertEqual({row["item_id"] for row in limited["records"]},
                             {item["id"] for item in self.data["items"][:4]})
            with self.assertRaises(ValueError):
                ow.capability_check(target, Fake(), limit=0)


@unittest.skipIf(np is None, "numpy optional extra is not installed")
class TextBaselineTests(unittest.TestCase):
    def test_vectors_are_deterministic_signed_and_unit_norm(self):
        first = ow._text_vector(np, "Gold on Dhanteras")
        np.testing.assert_array_equal(first, ow._text_vector(np, "Gold on Dhanteras"))
        self.assertAlmostEqual(float(np.linalg.norm(first)), 1.0, places=5)
        self.assertEqual(first.shape, (2 ** 14,))
        self.assertTrue((first < 0).any() and (first > 0).any())
        self.assertFalse(np.array_equal(first, ow._text_vector(np, "Silver on Dhanteras")))
        self.assertEqual(float(np.linalg.norm(ow._text_vector(np, "ab"))), 0.0)

    def test_kernel_form_matches_primal_ridge(self):
        prompts = ["gold festival offer one", "ordinary day schedule two", "gold festival offer three",
                   "ordinary day schedule four", "festival gold buying five", "plain weekday six",
                   "gold festival offer one", "unseen prompt text seven"]
        labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0])
        kernel, inverse = ow._text_kernel(np, prompts)
        features = np.stack([ow._text_vector(np, text) for text in prompts]).astype(np.float64)
        train, evaluate = np.asarray([0, 1, 2, 3, 4, 5]), np.asarray([6, 7, 0])
        decision = ow._kernel_decision(np, kernel, inverse, train, evaluate, labels, 0.7)
        coefficient, intercept = ow._ridge_fit(np, features[train], labels[train], 0.7)
        np.testing.assert_allclose(decision, features[evaluate] @ coefficient + intercept, atol=1e-4)

    def test_duplicate_prompts_are_vectorized_once(self):
        with patch.object(ow, "_text_vector", wraps=ow._text_vector) as spy:
            ow._text_cache.clear()
            ow._text_kernel(np, ["same prompt text", "same prompt text", "other prompt text"])
        self.assertEqual(spy.call_count, 2)

    def test_probe_vs_text_interpretation(self):
        worse = ow._probe_vs_text({"accuracy": 0.7}, {"accuracy": 0.9})
        equal = ow._probe_vs_text({"accuracy": 0.7}, {"accuracy": 0.7})
        better = ow._probe_vs_text({"accuracy": 0.9}, {"accuracy": 0.7})
        self.assertAlmostEqual(worse["difference"], -0.2)
        for report in (worse, equal):
            self.assertIn("condition decodability", report["interpretation"])
        self.assertAlmostEqual(better["difference"], 0.2)
        self.assertIn("condition decodability", better["interpretation"])


@unittest.skipIf(np is None, "numpy optional extra is not installed")
class ProbeInvariantTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "activations.npz"
        self.output = Path(self.temp.name) / "probe.npz"
        self.arrays = activation_fixture(self.source)

    def test_train_only_scaling_and_raw_direction_equivalence(self):
        report = ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            np.testing.assert_allclose(artifact["scaler_mean"], [0, 1, 7])
            np.testing.assert_allclose(artifact["scaler_scale"], [2, np.sqrt(2 / 3), 1])
            standardized = (self.arrays["features"] - artifact["scaler_mean"]) / artifact["scaler_scale"]
            raw_scores = self.arrays["features"] @ artifact["direction"] + artifact["intercept"]
            scaled_scores = standardized @ artifact["standardized_coefficient"] + artifact["standardized_intercept"]
            np.testing.assert_allclose(raw_scores, scaled_scores)
            metadata = json.loads(artifact["metadata"].item())
            self.assertEqual(metadata["activation_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        for split in ("train", "dev", "test"):
            self.assertEqual(report["splits"][split]["accuracy"], 1.0)
            self.assertEqual(report["splits"][split]["majority_baseline_accuracy"], 0.5)
            self.assertIn("cluster_label_flip_control", report["splits"][split])
        self.assertIn("not mediation", report["interpretation"])
        self.assertNotIn("all_tokens", json.dumps(report))

    def test_text_baseline_beats_chance_when_prompts_are_separable(self):
        report = ow.fit_probe(self.source, self.output)
        baseline = report["text_baseline"]
        for split in ("train", "dev", "test"):
            for key in ("accuracy", "cluster_macro_accuracy", "majority_baseline_accuracy", "balanced_accuracy"):
                self.assertIn(key, baseline["splits"][split])
        self.assertGreaterEqual(baseline["splits"]["test"]["accuracy"], 0.75)
        self.assertEqual(baseline["splits"]["test"]["n"], 8)
        comparison = report["probe_vs_text_baseline"]
        self.assertAlmostEqual(comparison["difference"], report["splits"]["test"]["accuracy"]
                               - baseline["splits"]["test"]["accuracy"])

    def test_text_baseline_is_chance_when_prompts_are_identical_across_classes(self):
        activation_fixture(self.source, prompts="identical")
        report = ow.fit_probe(self.source, self.output)
        for split in ("train", "dev", "test"):
            self.assertEqual(report["text_baseline"]["splits"][split]["accuracy"], 0.5)
        # The activation probe (separable features by construction) is clearly above that baseline.
        self.assertEqual(report["probe_vs_text_baseline"]["difference"], 0.5)
        self.assertIn("condition decodability", report["probe_vs_text_baseline"]["interpretation"])

    def test_leave_one_value_out_transfer(self):
        report = ow.fit_probe(self.source, self.output)
        transfer = report["transfer"]
        festival = transfer["festival"]
        self.assertEqual(sorted(festival["held_out"]), ["dhanteras", "muhurat"])
        for value in ("dhanteras", "muhurat"):
            row = festival["held_out"][value]
            self.assertEqual((row["n"], row["n_clusters"]), (4, 1))
            self.assertEqual(row["accuracy"], 1.0)
            self.assertIn("text_baseline_accuracy", row)
            self.assertTrue(0.0 <= row["text_baseline_accuracy"] <= 1.0)
        self.assertEqual(festival["macro_accuracy"], 1.0)
        self.assertEqual(festival["n_values_evaluated"], 2)
        language = transfer["language"]
        self.assertEqual(sorted(language["held_out"]), ["en", "hi"])
        for row in language["held_out"].values():
            self.assertEqual((row["n"], row["n_clusters"], row["accuracy"]), (4, 2, 1.0))
        self.assertEqual(language["note"], "cross-language transfer")
        self.assertAlmostEqual(language["macro_difference"],
                               language["macro_accuracy"] - language["macro_text_baseline_accuracy"])
        for field in ("template_id", "paraphrase_id"):
            self.assertIn("only 1 distinct", transfer[field]["skipped"])

    def test_template_transfer_runs_when_two_templates_are_present(self):
        activation_fixture(self.source, template_by_cluster=True)
        transfer = ow.fit_probe(self.source, self.output)["transfer"]
        self.assertEqual(sorted(transfer["template_id"]["held_out"]), ["t1", "t2"])
        self.assertEqual(transfer["template_id"]["macro_accuracy"], 1.0)

    def test_held_out_value_missing_a_class_is_skipped_with_reason(self):
        arrays = activation_fixture(self.source)
        trials = json.loads(arrays["trials"].item())
        for trial in trials:
            # Language fully confounded with condition: each language holds only one class.
            trial["language"] = "en" if trial["condition"] == "neutral" else "hi"
        arrays["trials"] = np.asarray(json.dumps(trials))
        metadata = json.loads(arrays["metadata"].item())
        metadata["trials_sha256"] = ow._trials_digest(trials)
        arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez(self.source, **arrays)
        language = ow.fit_probe(self.source, self.output)["transfer"]["language"]
        self.assertEqual(language["n_values_evaluated"], 0)
        self.assertIn("skipped", language)
        for row in language["held_out"].values():
            self.assertIn("lacks both", row["skipped"])

    def test_random_direction_null_matches_independent_recomputation(self):
        report = ow.fit_probe(self.source, self.output)
        null = report["random_direction_null"]
        self.assertEqual(null["n_directions"], 200)
        with np.load(self.output, allow_pickle=False) as artifact:
            coefficient = artifact["standardized_coefficient"]
            mean, scale = artifact["scaler_mean"], artifact["scaler_scale"]
        splits, labels = self.arrays["splits"], self.arrays["labels"]
        test = splits == "test"
        z = (self.arrays["features"][test] - mean) / scale
        directions = np.random.default_rng(ow._NULL_SEED).normal(size=(200, 3))
        directions *= np.linalg.norm(coefficient) / np.linalg.norm(directions, axis=1, keepdims=True)
        intercept = float((2.0 * labels[splits == "train"] - 1.0).mean())
        accuracy = np.asarray([(((z @ d + intercept) >= 0) == labels[test].astype(bool)).mean() for d in directions])
        self.assertAlmostEqual(null["accuracy_mean"], float(accuracy.mean()))
        self.assertAlmostEqual(null["accuracy_p5"], float(np.percentile(accuracy, 5)))
        self.assertAlmostEqual(null["accuracy_p95"], float(np.percentile(accuracy, 95)))
        self.assertAlmostEqual(null["fraction_ge_probe_accuracy"], float((accuracy >= 1.0 - 1e-12).mean()))
        self.assertEqual(null["probe_test_accuracy"], report["splits"]["test"]["accuracy"])
        self.assertLessEqual(null["accuracy_p5"], null["accuracy_p95"])
        self.assertIn("not a calibrated", null["interpretation"])

    def test_report_is_strict_json(self):
        report = ow.fit_probe(self.source, self.output)
        json.dumps(report, allow_nan=False)

    def test_no_cluster_leakage_even_with_distinct_trial_ids(self):
        self.arrays["cluster_ids"][self.arrays["splits"] == "test"] = "train-0"
        np.savez(self.source, **self.arrays)
        with self.assertRaisesRegex(ValueError, "Cluster leakage"):
            ow.fit_probe(self.source, self.output)
        self.assertFalse(self.output.exists())

    def test_reject_duplicate_trials_nonfinite_and_missing_class(self):
        mutations = (
            ("duplicate", lambda a: a["trial_ids"].__setitem__(1, a["trial_ids"][0])),
            ("nonfinite", lambda a: a["features"].__setitem__((0, 0), np.nan)),
            ("missing_class", lambda a: a["labels"].__setitem__(a["splits"] == "dev", 1)),
        )
        for name, change in mutations:
            with self.subTest(name=name):
                arrays = activation_fixture(self.source)
                change(arrays)
                np.savez(self.source, **arrays)
                with self.assertRaises(ValueError):
                    ow.fit_probe(self.source, self.output)

    def test_stored_trials_are_required_and_must_agree_with_labels(self):
        arrays = activation_fixture(self.source)
        without = {key: value for key, value in arrays.items() if key != "trials"}
        np.savez(self.source, **without)
        with self.assertRaisesRegex(ValueError, "stored trials"):
            ow.fit_probe(self.source, self.output)
        arrays["labels"][0] = 1 - arrays["labels"][0]
        np.savez(self.source, **arrays)
        with self.assertRaisesRegex(ValueError, "disagree"):
            ow.fit_probe(self.source, self.output)

    def test_heldout_labels_do_not_change_fitted_direction(self):
        ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            original = artifact["direction"].copy()
        heldout = self.arrays["splits"] != "train"
        self.arrays["labels"][heldout] = 1 - self.arrays["labels"][heldout]
        trials = json.loads(self.arrays["trials"].item())
        for trial, flag in zip(trials, heldout):
            if flag:
                trial["condition"] = "neutral" if trial["condition"] == "event" else "event"
        self.arrays["trials"] = np.asarray(json.dumps(trials))
        metadata = json.loads(self.arrays["metadata"].item())
        metadata["trials_sha256"] = ow._trials_digest(trials)
        self.arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez(self.source, **self.arrays)
        report = ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            np.testing.assert_array_equal(original, artifact["direction"])
        self.assertEqual(report["splits"]["test"]["accuracy"], 0.0)

    def test_dual_and_primal_ridge_agree_with_direct_solution(self):
        rng = np.random.default_rng(4)
        for n, width in ((8, 3), (5, 9)):
            x = rng.normal(size=(n, width))
            labels = np.arange(n) % 2
            coefficient, intercept = ow._ridge_fit(np, x, labels, 0.7)
            centered = x - x.mean(axis=0)
            target = 2 * labels - 1
            expected = np.linalg.solve(centered.T @ centered + 0.7 * np.eye(width),
                                       centered.T @ (target - target.mean()))
            np.testing.assert_allclose(coefficient, expected, atol=1e-12)
            self.assertAlmostEqual(intercept, target.mean() - x.mean(axis=0) @ expected)

    def test_mixed_valence_is_not_presented_as_auspiciousness(self):
        metadata = json.loads(self.arrays["metadata"].item())
        metadata["event_valences"] = ["auspicious", "avoidance"]
        self.arrays["metadata"] = np.asarray(json.dumps(metadata))
        np.savez(self.source, **self.arrays)
        with self.assertRaisesRegex(ValueError, "one event_valence"):
            ow.fit_probe(self.source, self.output)

    def test_provenance_mismatches_all_fail(self):
        ow.fit_probe(self.source, self.output)
        for field, value in (("model", "other"), ("revision", None),
                             ("resolved_revision", "new-commit"), ("local_model_sha256", "changed-bytes"), ("layer", 0),
                             ("width", 4), ("representation", "generated_answer")):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "provenance mismatch"):
                ow._read_steering(self.output, dict(provenance(), **{field: value}))

    def test_random_control_is_seeded_and_norm_matched(self):
        ow.fit_probe(self.source, self.output)
        learned = ow._read_steering(self.output, provenance())
        random1 = ow._read_steering(self.output, provenance(), 7)
        random2 = ow._read_steering(self.output, provenance(), 7)
        other = ow._read_steering(self.output, provenance(), 8)
        np.testing.assert_array_equal(random1, random2)
        self.assertFalse(np.allclose(random1, learned))
        self.assertFalse(np.allclose(random1, other))
        self.assertAlmostEqual(float(np.linalg.norm(random1)), float(np.linalg.norm(learned)))

    def test_probe_refuses_nonpositive_ridge(self):
        for ridge in (0, -1, float("nan")):
            with self.assertRaises(ValueError):
                ow.fit_probe(self.source, self.output, ridge=ridge)

    def test_embedded_trial_and_prompt_tampering_are_rejected(self):
        trials = json.loads(self.arrays["trials"].item())
        trials[0]["festival"] = "edited"
        self.arrays["trials"] = np.asarray(json.dumps(trials))
        np.savez(self.source, **self.arrays)
        with self.assertRaisesRegex(ValueError, "trial-content digest"):
            ow.fit_probe(self.source, self.output)
        trials[0]["prompt"] += " edited"
        self.arrays["trials"] = np.asarray(json.dumps(trials))
        np.savez(self.source, **self.arrays)
        with self.assertRaisesRegex(ValueError, "prompt hash"):
            ow.fit_probe(self.source, self.output)

    def test_calibrated_doses_match_diagonal_standardized_displacement(self):
        ow.fit_probe(self.source, self.output)
        with np.load(self.output, allow_pickle=False) as artifact:
            scale = artifact["scaler_scale"].copy()
        for seed in (None, 7, 9):
            direction = ow._read_steering(self.output, provenance(), seed, dose="activation_std")
            self.assertAlmostEqual(float(np.linalg.norm(direction / scale)), 1)

    def test_dev_probe_does_not_expose_test_scores_and_has_sensitivity(self):
        report = ow.fit_probe(self.source, self.output, evaluation_split="dev")
        self.assertNotIn("test", report["splits"])
        self.assertNotIn("test", report["text_baseline"]["splits"])
        self.assertEqual(report["random_direction_null"]["split"], "dev")
        self.assertEqual(report["transfer"]["evaluation_split"], "dev")
        self.assertEqual(len(report["training_cluster_sensitivity"]["refits"]), 3)
        self.assertEqual(report["text_baseline"]["selection"]["status"], "dev_cluster_macro_accuracy")
        self.assertIsNotNone(report["splits"]["dev"]["leave_one_cluster_out_macro_range"])


@unittest.skipIf(np is None, "numpy optional extra is not installed")
class RealDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
            from transformers import LlamaConfig, LlamaForCausalLM
        except ImportError as exc:
            raise unittest.SkipTest(f"Optional torch/transformers unavailable: {exc}")
        cls.torch = torch
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=2, num_attention_heads=2,
                             num_key_value_heads=2, max_position_embeddings=1024,
                             bos_token_id=1, eos_token_id=None, pad_token_id=0)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(31)
            cls.network = LlamaForCausalLM(config).eval()

        class CharacterTokenizer:
            chat_template = None
            pad_token_id = 0
            eos_token_id = None

            def __call__(self, text, **kwargs):
                ids = torch.tensor([[1] + [2 + ord(char) % 29 for char in text]])
                return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

            def decode(self, tokens, **kwargs):
                # Actual generated token IDs, deliberately not a fabricated task answer.
                return json.dumps(tokens.tolist())

        cls.tokenizer = CharacterTokenizer()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        patcher = patch.object(ow, "_load_runtime", return_value=(self.torch, self.tokenizer, self.network))
        self.load_mock = patcher.start()
        self.addCleanup(patcher.stop)

    def _steering_artifact(self):
        metadata = ow._provenance(self.network, "local-fixture", None, 1)
        metadata["kind"] = "ridge_condition_probe"
        direction = np.linspace(-0.4, 0.7, 16)
        path = self.root / "steer.npz"
        np.savez(path, direction=direction, metadata=np.asarray(json.dumps(metadata)))
        return path, direction

    def _assert_no_hooks(self):
        for layer in self.network.model.layers:
            self.assertEqual(len(layer._forward_hooks), 0)

    def test_real_residual_hook_signed_zero_and_tuple_preservation(self):
        torch = self.torch
        inputs = ow._inputs(self.tokenizer, "fixture", "cpu")
        block = self.network.model.layers[-1]
        captured = []
        capture = lambda module, args, output: captured.append(ow._hidden(output).detach().clone())
        hook = block.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                baseline_logits = self.network(**inputs).logits.clone()
        finally:
            hook.remove()
        baseline = captured.pop()
        direction = torch.arange(16, dtype=torch.float32) / 16
        for strength in (-2.0, 0.0, 2.0):
            steering = block.register_forward_hook(ow._ResidualSteering(direction, strength))
            capture_hook = block.register_forward_hook(capture)
            try:
                with torch.inference_mode():
                    logits = self.network(**inputs).logits
            finally:
                steering.remove()
                capture_hook.remove()
            shifted = captured.pop()
            torch.testing.assert_close(shifted, baseline + strength * direction)
            if strength == 0:
                torch.testing.assert_close(logits, baseline_logits, atol=0, rtol=0)
            else:
                self.assertFalse(torch.equal(logits, baseline_logits))
        marker = object()
        output = ow._ResidualSteering(direction, 1)(None, None, (baseline, marker))
        self.assertIs(output[1], marker)
        torch.testing.assert_close(output[0], baseline + direction)
        self.assertEqual(len(block._forward_hooks), 0)
        with self.assertRaises(ValueError):
            ow._ResidualSteering(direction, 1, "everywhere")

    def _prefill_and_decode_residuals(self, positions, strength):
        """Block outputs of the last layer for a real prefill then one cached decode step."""
        torch = self.torch
        inputs = ow._inputs(self.tokenizer, "abc", "cpu")
        block = self.network.model.layers[-1]
        direction = torch.arange(16, dtype=torch.float32) / 16 + 0.5

        def run(steered):
            captured = []
            hooks = []
            if steered:
                hooks.append(block.register_forward_hook(ow._ResidualSteering(direction, strength, positions)))
            hooks.append(block.register_forward_hook(
                lambda module, args, output: captured.append(ow._hidden(output).detach().clone())))
            try:
                with torch.inference_mode():
                    prefill = self.network(**inputs, use_cache=True)
                    self.network(input_ids=torch.tensor([[5]]), past_key_values=prefill.past_key_values, use_cache=True)
            finally:
                for hook in hooks:
                    hook.remove()
            return captured

        baseline, steered = run(False), run(True)
        self.assertEqual([c.shape[1] for c in baseline], [4, 1])
        return (steered[0] - baseline[0], steered[1] - baseline[1], direction * strength,
                steered[0], baseline[0], steered[1], baseline[1])

    def test_position_semantics_with_hook_captured_residuals(self):
        torch = self.torch
        # The steered block is the last decoder block, so earlier-layer state is unchanged and
        # prefill/decode residuals are comparable position by position with the baseline.
        d_pre, d_dec, shift, *_ = self._prefill_and_decode_residuals("all", 3.0)
        torch.testing.assert_close(d_pre, shift.expand_as(d_pre), atol=1e-5, rtol=0)
        torch.testing.assert_close(d_dec, shift.expand_as(d_dec), atol=1e-5, rtol=0)

        d_pre, d_dec, shift, *_ = self._prefill_and_decode_residuals("prompt_last", 3.0)
        self.assertTrue(torch.equal(d_pre[:, :-1], torch.zeros_like(d_pre[:, :-1])))
        torch.testing.assert_close(d_pre[:, -1], shift.expand_as(d_pre[:, -1]), atol=1e-5, rtol=0)
        self.assertTrue(torch.equal(d_dec, torch.zeros_like(d_dec)))

        _, d_dec, shift, pre_s, pre_b, *_ = self._prefill_and_decode_residuals("generated", 3.0)
        self.assertTrue(torch.equal(pre_s, pre_b))
        torch.testing.assert_close(d_dec, shift.expand_as(d_dec), atol=1e-5, rtol=0)

    def _generate_with_capture(self, provider, prompt="abc"):
        captured = []
        hook = self.network.model.norm.register_forward_pre_hook(
            lambda module, args: captured.append(args[0].detach().clone()))
        try:
            result = provider.generate(prompt)
        finally:
            hook.remove()
        return result, captured

    def test_provider_applies_selected_positions_during_generation(self):
        torch = self.torch
        path, direction = self._steering_artifact()
        shift = 3.0 * torch.as_tensor(direction, dtype=torch.float32)
        plain = ow.OpenWeightProvider("local-fixture", max_output_tokens=2)
        _, baseline = self._generate_with_capture(plain)
        self.assertEqual([c.shape[1] for c in baseline], [4, 1])
        for position in ("all", "prompt_last", "generated"):
            provider = ow.OpenWeightProvider("local-fixture", max_output_tokens=2, steering_path=path,
                                             strength=3.0, steering_positions=position)
            result, steered = self._generate_with_capture(provider)
            self.assertEqual(result["generation"]["steering_positions"], position)
            self.assertEqual([c.shape[1] for c in steered], [4, 1])
            prefill = steered[0] - baseline[0]
            if position == "all":
                torch.testing.assert_close(prefill, shift.expand_as(prefill), atol=1e-5, rtol=0)
            elif position == "prompt_last":
                self.assertTrue(torch.equal(prefill[:, :-1], torch.zeros_like(prefill[:, :-1])))
                torch.testing.assert_close(prefill[:, -1], shift.expand_as(prefill[:, -1]), atol=1e-5, rtol=0)
            else:
                # Unchanged prefill logits give the same first token, so the decode step is comparable.
                self.assertTrue(torch.equal(steered[0], baseline[0]))
                decode = steered[1] - baseline[1]
                torch.testing.assert_close(decode, shift.expand_as(decode), atol=1e-5, rtol=0)
            self._assert_no_hooks()

    def test_zero_strength_is_exactly_baseline_for_every_position(self):
        path, _ = self._steering_artifact()
        plain = ow.OpenWeightProvider("local-fixture", max_output_tokens=3)
        baseline = plain.generate("abc")
        for position in ("all", "prompt_last", "generated"):
            zero = ow.OpenWeightProvider("local-fixture", max_output_tokens=3, steering_path=path,
                                         strength=0, steering_positions=position)
            self.assertEqual(zero.generate("abc")["raw_text"], baseline["raw_text"])
        self._assert_no_hooks()

    def test_provider_generates_and_zero_control_matches_baseline(self):
        path, _ = self._steering_artifact()
        plain = ow.OpenWeightProvider("local-fixture", max_output_tokens=3)
        zero = ow.OpenWeightProvider("local-fixture", max_output_tokens=3, steering_path=path, strength=0)
        before = self.torch.random.get_rng_state().clone()
        baseline, control = plain.generate("abc"), zero.generate("abc")
        self.assertEqual(baseline["raw_text"], control["raw_text"])
        self.assertEqual(baseline["usage"], {"input_tokens": 4, "output_tokens": 3})
        self.assertEqual(baseline["status"], "invalid")
        self.assertEqual(baseline["generation"]["resolved_layer"], 1)
        self.torch.testing.assert_close(before, self.torch.random.get_rng_state())
        self.assertEqual(len(self.network.model.layers[1]._forward_hooks), 0)

    def test_generate_text_is_unvalidated_and_keyed_by_seed(self):
        provider = ow.OpenWeightProvider("local-fixture", temperature=0.8, max_output_tokens=3, seed=5)
        first = provider.generate_text("abc", key="capability:en:c01")
        again = provider.generate_text("abc", key="capability:en:c01")
        other = provider.generate_text("abc", key="capability:hi:c01")
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["raw_text"], again["raw_text"])
        self.assertNotEqual(first["sampling_seed"], other["sampling_seed"])

    def test_fast_tokenizer_segment_ids_do_not_break_causal_generation(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast
        backend = Tokenizer(WordLevel({"[UNK]": 0, "financial": 1, "decision": 2}, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]")
        prompt = "financial decision"
        with self.torch.inference_mode():
            expected = self.network.generate(
                **tokenizer(prompt, return_tensors="pt", return_token_type_ids=False),
                max_new_tokens=3, do_sample=False, pad_token_id=0,
            )
        with patch.object(ow, "_load_runtime", return_value=(self.torch, tokenizer, self.network)):
            result = ow.OpenWeightProvider("local-fixture", max_output_tokens=3).generate(prompt)
        self.assertEqual(result["raw_text"], tokenizer.decode(expected[0, 2:], skip_special_tokens=True))
        self.assertEqual(result["usage"]["input_tokens"], 2)
        self.assertEqual(result["status"], "invalid")

    def test_generation_failure_removes_hook_for_every_position(self):
        path, _ = self._steering_artifact()
        for position in ("all", "prompt_last", "generated"):
            provider = ow.OpenWeightProvider("local-fixture", steering_path=path, strength=-1,
                                             steering_positions=position)
            with patch.object(self.network, "generate", side_effect=RuntimeError("intentional failure")):
                with self.assertRaisesRegex(RuntimeError, "intentional failure"):
                    provider.generate("abc")
            self._assert_no_hooks()

    def test_changed_artifact_is_not_used(self):
        path, _ = self._steering_artifact()
        provider = ow.OpenWeightProvider("local-fixture", steering_path=path)
        path.write_bytes(b"changed artifact")
        with self.assertRaisesRegex(ValueError, "changed after"):
            provider.generate("abc")

    def test_trial_sampling_seeds_differ_and_reproduce_without_splitting_settings(self):
        provider = ow.OpenWeightProvider("local-fixture", temperature=0.8, max_output_tokens=3, seed=19)
        first = provider.generate_trial({"trial_id": "replicate-1", "prompt": "abc"})
        second = provider.generate_trial({"trial_id": "replicate-2", "prompt": "abc"})
        replay = provider.generate_trial({"trial_id": "replicate-1", "prompt": "abc"})
        self.assertNotEqual(first["sampling_seed"], second["sampling_seed"])
        self.assertEqual(first["sampling_seed"], replay["sampling_seed"])
        self.assertEqual(first["raw_text"], replay["raw_text"])
        self.assertEqual(first["generation"], second["generation"])
        self.assertEqual(first["generation"]["seed"], 19)
        self.assertIn("steering_sha256", first["generation"])

    def test_extraction_is_prompt_only_matches_final_token_and_skips_controls(self):
        from auspicious.core import write_jsonl
        trials = [v2_trial("event", "festival fixture"), v2_trial("neutral", "ordinary fixture")]
        control = v2_trial("no_calendar", "unlabelled fixture")
        source, output = self.root / "trials.jsonl", self.root / "activations.npz"
        write_jsonl(source, trials + [control])
        with patch.object(self.network, "generate", side_effect=AssertionError("Extraction must not generate")):
            report = ow.extract_activations(source, output, model="local-fixture", layer=-1)
        self.assertEqual(report["layer"], 1)
        self.assertEqual(report["n_trials"], 2)
        self.assertEqual(report["n_control_trials_excluded"], 1)
        expected = []
        hook = self.network.model.layers[1].register_forward_hook(
            lambda module, args, output: expected.append(ow._hidden(output)[0, -1].detach().clone())
        )
        try:
            with self.torch.inference_mode():
                for trial in trials:
                    self.network(**ow._inputs(self.tokenizer, trial["prompt"], "cpu"), use_cache=False)
        finally:
            hook.remove()
        with np.load(output, allow_pickle=False) as artifact:
            np.testing.assert_allclose(artifact["features"], self.torch.stack(expected).numpy())
            np.testing.assert_array_equal(artifact["labels"], [1, 0])
            self.assertEqual(json.loads(artifact["trials"].item()), trials)
            self.assertEqual(json.loads(artifact["metadata"].item())["trials_sha256"], ow._trials_digest(trials))
        self.assertEqual(len(self.network.model.layers[1]._forward_hooks), 0)

    def test_capability_check_on_real_provider_records_identity_and_invalid_rate(self):
        path, _ = self._steering_artifact()
        provider = ow.OpenWeightProvider("local-fixture", max_output_tokens=3, steering_path=path,
                                         strength=1.0, steering_positions="generated")
        report = ow.capability_check(self.root / "capability.json", provider, limit=2)
        self.assertEqual(report["overall"]["n"], 10)
        # The random tiny model emits token-id lists, never the requested JSON object.
        self.assertEqual(report["overall"]["invalid_rate"], 1.0)
        self.assertEqual(report["overall"]["accuracy"], 0.0)
        identity = report["provider_identity"]["generation"]
        self.assertEqual((identity["steering_positions"], identity["strength"], identity["resolved_layer"]),
                         ("generated", 1.0, 1))
        self._assert_no_hooks()

    def _grid_corpus(self, valences=("auspicious",)):
        from auspicious.core import write_jsonl
        trials = []
        plan = [("train", 3), ("dev", 2), ("test", 2)]
        index = 0
        for split, count in plan:
            for n in range(count):
                cluster = f"{split}-{n}"
                pair = f"pair-{cluster}"
                valence = valences[index % len(valences)]
                index += 1
                for condition, prompt in (("event", f"festival gold offer {cluster}"),
                                          ("neutral", f"ordinary day plan {cluster}"),
                                          ("no_calendar", f"plain plan {cluster}")):
                    trial = v2_trial(condition, prompt, pair=pair, cluster=cluster, split=split)
                    trial["event_valence"] = valence
                    trials.append(trial)
        path = self.root / f"grid-trials-{'-'.join(valences)}.jsonl"
        write_jsonl(path, trials)
        return path, trials

    def test_steering_grid_manifest_resume_and_dev_restriction(self):
        source, trials = self._grid_corpus()
        out = self.root / "grid"
        kwargs = dict(model="local-fixture", layers=[0, -1], strengths=[1.0], positions=("all", "generated"),
                      random_seeds=(7,), max_output_tokens=3, capability_limit=2)
        manifest = ow.steering_grid(source, out, **kwargs)
        self.assertTrue(manifest["complete"])
        self.assertFalse(manifest["test_only"])
        self.assertEqual(manifest["evaluation_split"], "dev")
        self.assertEqual(manifest["settings"]["layers_resolved"], [0, 1])
        # per layer: baseline + 2 positions x (learned + 1 random seed)
        self.assertEqual(manifest["n_runs"], 2 * (1 + 2 * 2))
        kinds = sorted(run["kind"] for run in manifest["runs"])
        self.assertEqual(kinds, ["baseline"] * 2 + ["learned"] * 4 + ["random"] * 4)
        test_trials = [t for t in trials if t["split"] == "dev"]
        self.assertEqual(len(test_trials), 6)
        for run in manifest["runs"]:
            self.assertEqual(sum(run["counts"].values()), len(test_trials))
            records = [json.loads(line) for line in (Path(run["dir"]) / "records.jsonl").read_text().splitlines()]
            self.assertEqual({r["trial"]["split"] for r in records}, {"dev"})
            self.assertEqual({r["trial"]["condition"] for r in records}, {"event", "neutral", "no_calendar"})
            capability = json.loads((Path(run["dir"]) / "capability_check.json").read_text())
            generation = capability["provider_identity"]["generation"]
            self.assertEqual(generation["steering_positions"], run["position"])
            self.assertEqual(generation["strength"], run["strength"])
            self.assertEqual(generation["random_direction_seed"], run["random_seed"])
            self.assertEqual(run["capability"]["overall"]["n"], 10)
            self.assertEqual(run["executed"], len(test_trials))
        self.assertEqual({Path(run["dir"]).relative_to(out).as_posix() for run in manifest["runs"] if run["layer"] == 1},
                         {"L1/baseline", "L1/all/s1", "L1/generated/s1",
                          "L1/all/random7/s1", "L1/generated/random7/s1"})
        # The probe is trained without test clusters and the activations contain only event/neutral rows.
        with np.load(out / "L1" / "activations.npz", allow_pickle=False) as artifact:
            self.assertEqual(len(artifact["labels"]), 14)
        self.assertTrue(all(not info["reused"] for info in manifest["probes"].values()))
        self.assertEqual(json.loads((out / "grid.json").read_text())["n_runs"], manifest["n_runs"])
        before = {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*.npz")}
        capability_bytes = (out / "L1" / "all" / "s1" / "capability_check.json").read_bytes()
        again = ow.steering_grid(source, out, **kwargs)
        self.assertTrue(all(run["executed"] == 0 for run in again["runs"]))
        self.assertTrue(all(info["reused"] for info in again["probes"].values()))
        self.assertEqual(before, {path.relative_to(out).as_posix(): path.read_bytes() for path in out.rglob("*.npz")})
        self.assertEqual(capability_bytes, (out / "L1" / "all" / "s1" / "capability_check.json").read_bytes())
        self._assert_no_hooks()

    def test_steering_grid_refreshes_capability_after_item_changes(self):
        source, _ = self._grid_corpus()
        out, items = self.root / "grid", self.root / "items.json"
        data, _, _ = ow._load_capability(None)
        items.write_text(json.dumps(data), encoding="utf-8")
        kwargs = dict(model="local-fixture", layers=[0], strengths=[1.0], max_output_tokens=1,
                      capability_items_path=str(items), capability_limit=1)
        ow.steering_grid(source, out, **kwargs)
        report_path = out / "L0" / "baseline" / "capability_check.json"
        first = json.loads(report_path.read_text())
        data["languages"]["en"]["instruction"] += " Updated capability task."
        items.write_text(json.dumps(data), encoding="utf-8")
        resumed = ow.steering_grid(source, out, **kwargs)
        refreshed = json.loads(report_path.read_text())
        self.assertNotEqual(first["items_sha256"], refreshed["items_sha256"])
        self.assertEqual(refreshed["items_sha256"], hashlib.sha256(items.read_bytes()).hexdigest())
        self.assertTrue(all(run["executed"] == 0 for run in resumed["runs"]))
        self._assert_no_hooks()

    def test_test_behavior_requires_one_frozen_dev_selection(self):
        source, _ = self._grid_corpus()
        kwargs = dict(model="local-fixture", layers=[0], strengths=[1.0], max_output_tokens=1, capability_limit=1)
        with self.assertRaisesRegex(ValueError, "frozen dev selection"):
            ow.steering_grid(source, self.root / "test", evaluation_split="test", **kwargs)
        dev = ow.steering_grid(source, self.root / "dev", **kwargs)
        self.assertTrue(dev["controls_complete"])
        self.assertIn("paired_vs_baseline", next(r for r in dev["runs"] if r["kind"] == "learned"))
        self.assertIn("paired_vs_learned", next(r for r in dev["runs"] if r["kind"] == "random"))
        selected = self.root / "selection.json"
        ow.freeze_steering_selection(dev["output"], selected, layer=0, position="all", strength=1.0)
        test = ow.steering_grid(source, self.root / "test", evaluation_split="test", frozen_selection=selected, **kwargs)
        self.assertTrue(test["test_only"])
        for row in test["runs"]:
            records = [json.loads(line) for line in (Path(row["dir"]) / "records.jsonl").read_text().splitlines()]
            self.assertEqual({r["trial"]["split"] for r in records}, {"test"})
        with self.assertRaisesRegex(ValueError, "exactly the frozen"):
            ow.steering_grid(source, self.root / "multi", evaluation_split="test", frozen_selection=selected,
                             **{**kwargs, "strengths": [1.0, 2.0]})
        selected_data = json.loads(selected.read_text())
        selected_data["probe_sha256"] = "changed"
        selected.write_text(json.dumps(selected_data))
        with self.assertRaisesRegex(ValueError, "artifact hash"):
            ow.steering_grid(source, self.root / "tampered", evaluation_split="test", frozen_selection=selected, **kwargs)

    def test_uncontrolled_grid_is_flagged_and_cannot_be_frozen(self):
        source, _ = self._grid_corpus()
        dev = ow.steering_grid(source, self.root / "uncontrolled", model="local-fixture", layers=[0],
                               strengths=[1.0], random_seeds=(), capability=False, max_output_tokens=1)
        self.assertFalse(dev["controls_complete"])
        with self.assertRaisesRegex(ValueError, "random controls"):
            ow.freeze_steering_selection(dev["output"], self.root / "selection.json", layer=0, position="all", strength=1.0)

    def test_steering_grid_rejects_bad_inputs_before_loading_a_model(self):
        source, _ = self._grid_corpus()
        mixed, _ = self._grid_corpus(valences=("auspicious", "avoidance"))
        base = dict(model="local-fixture", layers=[0], strengths=[1.0])
        with self.assertRaisesRegex(ValueError, "one event_valence"):
            ow.steering_grid(mixed, self.root / "mixed", **base)
        for change in ({"positions": ("sideways",)}, {"positions": ("all", "all")}, {"strengths": [1, 1.0]},
                       {"layers": []}, {"random_seeds": (1, 1)}, {"random_seeds": (-1,)}, {"ridge": 0},
                       {"layers": "0"}, {"strengths": [float("nan")]}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                ow.steering_grid(source, self.root / "bad", **dict(base, **change))
        self.load_mock.assert_not_called()
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ow.steering_grid(source, self.root / "duplicate", model="local-fixture", layers=[1, -1], strengths=[1.0])


if __name__ == "__main__":
    unittest.main()
