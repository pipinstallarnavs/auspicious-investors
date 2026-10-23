"""Synthetic budget and consistent-checkpoint checks, with no cloud requests."""
import json
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from auspicious.cloud import Checkpoint, estimate_cost, run_cloud
from auspicious.core import read_jsonl, run_trials, write_jsonl
from auspicious.gemini import GeminiProvider
from test_runtime import CountingProvider, payload, trials

try:
    from google.cloud import storage
except ImportError:
    storage = None


class Blob:
    def __init__(self):
        self.data = None
        self.writes = 0
        self.generation = 0

    def upload_from_filename(self, path):
        self.data = Path(path).read_bytes()
        self.writes += 1

    def upload_from_string(self, data, **kwargs):
        self.data = data
        self.writes += 1
        self.generation += 1

    def exists(self):
        return self.data is not None

    def download_as_bytes(self):
        return self.data.encode() if isinstance(self.data, str) else self.data

    def download_as_text(self):
        return self.download_as_bytes().decode()

    def download_to_filename(self, path):
        Path(path).write_bytes(self.download_as_bytes())

    def delete(self, **kwargs):
        self.data = None


class Bucket:
    def __init__(self):
        self.blobs = {}

    def blob(self, name):
        return self.blobs.setdefault(name, Blob())


class CloudTests(unittest.TestCase):
    def specification(self):
        spec = json.loads(Path("cloud/run-template.json").read_text())
        spec.update(project="research-project", design_config="examples/pilot-controls.json", bootstrap=0)
        return spec

    def test_bad_or_expired_cost_assumptions_are_rejected_before_requests(self):
        for value in (-1, 0, float("nan"), float("inf"), True):
            spec = self.specification()
            spec["model_budget_usd"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                estimate_cost([], spec)
        spec = self.specification()
        spec["pricing_valid_until"] = "2000-01-01"
        with self.assertRaisesRegex(ValueError, "pricing has expired"):
            estimate_cost([], spec)

    def test_cost_plan_reserves_max_output_and_all_attempts_without_authentication(self):
        spec = {"project": "research-project", "model": "gemini-3.8-flash", "model_budget_usd": 90,
                "generation": {"max_output_tokens": 1000, "max_attempts": 2, "thinking_level": "LOW"},
                "prices_usd_per_million": {"input": 0.75, "output": 3.75}, "pricing_source": "fixture"}
        plan = estimate_cost([{"prompt": "हिन्दी"}, {"prompt": "English"}], spec)
        self.assertEqual(plan["max_output_tokens_per_attempt"], 2000)
        self.assertGreater(plan["conservative_input_tokens_per_attempt"], 2048)
        expected = 2 * (plan["conservative_input_tokens_per_attempt"] * 0.75 + 2000 * 3.75) / 1e6
        self.assertEqual(plan["model_cost_upper_bound_usd"], expected)

    def test_sqlite_checkpoint_captures_wal_and_resumes_without_repeating_results(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "trials.jsonl"
            write_jsonl(source, trials())
            bucket = Bucket()
            checkpoint = Checkpoint(bucket, "runs/fixture", interval=0)
            run_trials(source, root / "first", CountingProvider(), checkpoint=checkpoint)
            snapshot = bucket.blob("runs/fixture/checkpoint.sqlite")
            self.assertGreaterEqual(snapshot.writes, 4)
            resumed = root / "resumed"
            resumed.mkdir()
            (resumed / "journal.sqlite").write_bytes(snapshot.data)
            provider = CountingProvider()
            summary = run_trials(source, resumed, provider)
            self.assertEqual((provider.calls, summary["stored"], summary["executed"]), (0, 2, 0))
            self.assertEqual(len(read_jsonl(resumed / "records.jsonl")), 2)
            progress = json.loads(bucket.blob("runs/fixture/progress.json").data)
            self.assertEqual(progress["stored"], 2)

    def test_checkpoint_throttling_and_forced_flush(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bucket = Bucket()
            checkpoint = Checkpoint(bucket, "runs/fixture", interval=float("inf"))
            with sqlite3.connect(root / "journal.sqlite") as db:
                db.execute("CREATE TABLE results (status TEXT)")
                db.commit()
                checkpoint(db, root, force=True)
                db.execute("INSERT INTO results VALUES ('ok')")
                db.commit()
                checkpoint(db, root)
                self.assertEqual(bucket.blob("runs/fixture/checkpoint.sqlite").writes, 1)
                checkpoint(db, root, force=True)
                self.assertEqual(bucket.blob("runs/fixture/checkpoint.sqlite").writes, 2)

    @unittest.skipIf(storage is None, "Google Cloud optional dependencies are not installed")
    def test_cloud_worker_publishes_complete_archive_and_is_idempotent(self):
        bucket = Bucket()
        spec = self.specification()
        bucket.blob("runs/fixture/run.json").upload_from_string(json.dumps(spec))
        reply = {"status": "ok", "raw_text": json.dumps(payload()), "usage": {"totalTokenCount": 10}}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(storage, "Client", return_value=SimpleNamespace(bucket=lambda _: bucket)), \
             patch.object(GeminiProvider, "prepare"), \
             patch.object(GeminiProvider, "generate_trial", return_value=reply) as generate:
            state = run_cloud("bucket", "fixture", directory=folder)
            self.assertEqual((state["state"], state["summary"]["stored"]), ("completed", 10))
            self.assertTrue(bucket.blob("runs/fixture/results.zip").exists())
            self.assertTrue(bucket.blob("runs/fixture/analysis/effects.csv").exists())
            self.assertFalse(bucket.blob("runs/fixture/lease.json").exists())
            run_cloud("bucket", "fixture", directory=folder)
            self.assertEqual(generate.call_count, 10)
            spec["generation"]["seed"] += 1
            bucket.blob("runs/fixture/run.json").upload_from_string(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "specification changed"):
                run_cloud("bucket", "fixture", directory=folder)

    @unittest.skipIf(storage is None, "Google Cloud optional dependencies are not installed")
    def test_cloud_worker_refuses_an_over_budget_run_without_model_calls(self):
        bucket = Bucket()
        spec = self.specification()
        spec["model_budget_usd"] = 0.0000001
        bucket.blob("runs/fixture/run.json").upload_from_string(json.dumps(spec))
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(storage, "Client", return_value=SimpleNamespace(bucket=lambda _: bucket)), \
             patch.object(GeminiProvider, "generate_trial") as generate:
            with self.assertRaisesRegex(ValueError, "exceeds this run"):
                run_cloud("bucket", "fixture", directory=folder)
            generate.assert_not_called()
            self.assertEqual(json.loads(bucket.blob("runs/fixture/status.json").data)["state"], "failed")
            self.assertFalse(bucket.blob("runs/fixture/lease.json").exists())


if __name__ == "__main__":
    unittest.main()
