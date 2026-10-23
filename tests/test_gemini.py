"""Offline API transport checks; no credentials or paid requests are used."""
import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from auspicious.gemini import GeminiProvider


class Credentials:
    valid = True
    token = "TEST_OAUTH_SECRET"


def response():
    return io.BytesIO(json.dumps({"modelVersion": "fixture-version", "usageMetadata": {"totalTokenCount": 12},
        "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"text": "hidden", "thought": True}, {"text": '{"fixture": true}'}]}}]}).encode())


class GeminiTransportTests(unittest.TestCase):
    def test_vertex_uses_adc_bearer_auth_and_records_backend_without_secrets(self):
        credentials = Credentials()
        provider = GeminiProvider("gemini-3.8-flash", project="research-project", credentials=credentials,
                                  thinking_level="LOW", seed=42, rpm=60000)
        with patch("auspicious.gemini.load_keys", side_effect=AssertionError("API key requested")), \
             patch("urllib.request.urlopen", side_effect=lambda *a, **k: response()) as opened:
            result = provider.generate_trial({"trial_id": "one", "prompt": "Synthetic task"})
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://aiplatform.googleapis.com/v1/projects/research-project/"
                         "locations/global/publishers/google/models/gemini-3.8-flash:generateContent")
        self.assertEqual(request.get_header("Authorization"), "Bearer TEST_OAUTH_SECRET")
        self.assertIsNone(request.get_header("X-goog-api-key"))
        self.assertEqual(result["provider"], "vertex_ai")
        self.assertEqual(result["model_version"], "fixture-version")
        self.assertEqual(result["raw_text"], '{"fixture": true}')
        self.assertEqual(provider.identity["project"], "research-project")
        self.assertNotIn("TEST_OAUTH_SECRET", json.dumps([provider.identity, result]))
        self.assertEqual(json.loads(request.data)["generationConfig"]["thinkingConfig"], {"thinkingLevel": "LOW"})

    def test_vertex_regional_endpoint_and_token_rotation(self):
        credentials = Credentials()
        provider = GeminiProvider("fixture", project="research-project", location="us-central1",
                                  credentials=credentials, rpm=60000)
        with patch("urllib.request.urlopen", side_effect=lambda *a, **k: response()) as opened:
            provider.generate("first")
            credentials.token = "ROTATED_SECRET"
            provider.generate("second")
        self.assertTrue(opened.call_args.args[0].full_url.startswith("https://us-central1-aiplatform.googleapis.com/"))
        self.assertEqual(opened.call_args.args[0].get_header("Authorization"), "Bearer ROTATED_SECRET")

    def test_api_key_transport_remains_supported_and_retry_keeps_the_same_key(self):
        error = urllib.error.HTTPError("https://example.test", 429, "quota", {}, None)
        with patch.dict(os.environ, {"GEMINI_API_KEYS": '["TEST_KEY_ONE","TEST_KEY_TWO"]'}):
            provider = GeminiProvider("fixture", rpm=60000, max_attempts=2)
        with patch.object(provider._quota, "acquire"), patch.object(provider._quota, "penalize"), \
             patch("urllib.request.urlopen", side_effect=[error, response()]) as opened:
            result = provider.generate("Synthetic task")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual([c.args[0].get_header("X-goog-api-key") for c in opened.call_args_list],
                         ["TEST_KEY_ONE", "TEST_KEY_ONE"])
        self.assertNotIn("TEST_KEY", json.dumps(result))

    def test_bad_cloud_identifiers_and_unsupported_thinking_are_rejected(self):
        for options in ({"project": "../../other"}, {"project": "research-project", "location": "../global"},
                        {"project": "research-project", "thinking_level": "MINIMAL"}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GeminiProvider("gemini-3.8-flash", **options)


if __name__ == "__main__":
    unittest.main()
