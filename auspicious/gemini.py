"""Gemini REST transport with bounded shared quota handling and no stored secrets."""
from __future__ import annotations

import email.utils
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request

from .core import RESPONSE_SCHEMA, strict_json

API_BASE = "https://generativelanguage.googleapis.com/v1beta"


def load_keys():
    """GEMINI_API_KEYS is a JSON string array; never put keys in experiment files."""
    multiple = os.environ.get("GEMINI_API_KEYS")
    if multiple:
        try:
            keys = strict_json(multiple)
        except ValueError as exc:
            raise ValueError("GEMINI_API_KEYS must be a JSON array of secret strings") from exc
        if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
            raise ValueError("GEMINI_API_KEYS must be a nonempty JSON array of secret strings")
    else:
        single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        keys = [single] if single else []
    if not keys:
        raise ValueError("No credentials: set GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEYS (JSON array)")
    if any("\n" in key or "\r" in key for key in keys):
        raise ValueError("API key contains invalid header characters")
    return tuple(dict.fromkeys(k.strip() for k in keys))


class _QuotaGate:
    def __init__(self, rpm):
        self.interval = 60.0 / rpm
        self.next_start = 0.0
        self.cooldown = 0.0
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                wait = max(self.next_start, self.cooldown) - now
                if wait <= 0:
                    self.next_start = now + self.interval
                    return
            time.sleep(wait)

    def penalize(self, seconds):
        with self.lock:
            self.cooldown = max(self.cooldown, time.monotonic() + seconds)


def _retry_after(headers):
    value = headers.get("Retry-After")
    if not value:
        return 0.0
    try:
        number = float(value)
        return max(0.0, number) if math.isfinite(number) else 0.0
    except ValueError:
        try:
            return max(0.0, email.utils.parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return 0.0


class GeminiProvider:
    def __init__(self, model, *, temperature=0.7, max_output_tokens=1024,
                 seed=None, rpm=30.0, timeout=120.0, max_attempts=4,
                 project=None, location="global", thinking_level=None, credentials=None):
        model = model.removeprefix("models/")
        if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
            raise ValueError("Invalid Gemini model identifier")
        if not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be finite and in [0,2]")
        if not math.isfinite(rpm) or rpm <= 0 or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("rpm and timeout must be positive finite numbers")
        if max_output_tokens < 1 or max_attempts < 1:
            raise ValueError("max_output_tokens and max_attempts must be positive")
        if project is not None and not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
            raise ValueError("Invalid Google Cloud project identifier")
        if not re.fullmatch(r"[a-z][a-z0-9-]*", location):
            raise ValueError("Invalid Vertex AI location")
        if thinking_level is not None and thinking_level not in {"MINIMAL", "LOW", "MEDIUM", "HIGH"}:
            raise ValueError("thinking_level must be MINIMAL, LOW, MEDIUM or HIGH")
        if model == "gemini-3.8-flash" and thinking_level == "MINIMAL":
            raise ValueError("Gemini 3.8 Flash requires LOW, MEDIUM or HIGH thinking")
        self.project, self.location = project, location
        self._credentials = credentials
        self._auth_lock = threading.Lock()
        self._keys = () if project else load_keys()
        self._key_index = 0
        self._key_lock = threading.Lock()
        self._quota = _QuotaGate(rpm)
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.generation = {"temperature": temperature, "maxOutputTokens": max_output_tokens,
                           "responseMimeType": "application/json", "responseSchema": RESPONSE_SCHEMA}
        if thinking_level is not None:
            self.generation["thinkingConfig"] = {"thinkingLevel": thinking_level}
        if seed is not None:
            self.generation["seed"] = seed
        self.identity = {"provider": "vertex_ai" if project else "gemini", "requested_model": model,
                         "generation": self.generation,
                         "transport": "vertex-ai-rest-v1-adc" if project else "gemini-rest-v1beta",
                         "quota_policy": "shared global gate across keys; retry same key on 429",
                         "rpm": rpm, "timeout": timeout, "max_attempts": max_attempts}
        if project:
            self.identity.update(project=project, location=location)

    def prepare(self):
        if self.project and self._credentials is None:
            import google.auth
            self._credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

    def _request(self, body, key):
        headers = {"Content-Type": "application/json", "User-Agent": "auspicious-investors/0.1.0"}
        if self.project:
            self.prepare()
            with self._auth_lock:
                if not self._credentials.valid:
                    from google.auth.transport.requests import Request
                    self._credentials.refresh(Request())
                headers["Authorization"] = "Bearer " + self._credentials.token
            host = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
            url = (f"https://{host}/v1/projects/{self.project}/locations/{self.location}/"
                   f"publishers/google/models/{self.model}:generateContent")
        else:
            headers["x-goog-api-key"] = key
            url = f"{API_BASE}/models/{self.model}:generateContent"
        return urllib.request.Request(url, data=body, headers=headers, method="POST")

    def generate_trial(self, trial):
        import hashlib
        base = self.generation.get("seed")
        sample_seed = None if base is None else int.from_bytes(
            hashlib.sha256(f"{base}:{trial['trial_id']}".encode()).digest()[:4], "big") % (2 ** 31)
        return self._generate(trial["prompt"], sample_seed)

    def generate(self, prompt):
        return self._generate(prompt, self.generation.get("seed"))

    def _generate(self, prompt, sample_seed):
        key = None
        if self._keys:
            with self._key_lock:
                slot = self._key_index % len(self._keys)
                key = self._keys[slot]
                self._key_index += 1
        generation = dict(self.generation)
        if sample_seed is not None:
            generation["seed"] = sample_seed
        body = json.dumps({"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                           "generationConfig": generation}, ensure_ascii=False).encode()
        result = {"provider": self.identity["provider"], "requested_model": self.model,
                  "model_version": None, "generation": self.generation,
                  "usage": {}, "attempts": 0, "status": "error", "raw_text": "",
                  "payload": None, "error": None, "attempt_log": [], "sampling_seed": sample_seed}
        for attempt in range(1, self.max_attempts + 1):
            self._quota.acquire()
            result["attempts"] = attempt
            request = self._request(body, key)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                api = strict_json(raw)
                if not isinstance(api, dict):
                    raise ValueError("Non-object API response")
                result["attempt_log"].append({"attempt": attempt, "http_status": 200})
                result["raw_response"] = api
                result["model_version"] = api.get("modelVersion")
                result["usage"] = api.get("usageMetadata", {})
                result["response_id"] = api.get("responseId")
                candidates = api.get("candidates", [])
                blocked = api.get("promptFeedback", {}).get("blockReason")
                if blocked:
                    result.update(status="blocked", error=f"Prompt blocked: {blocked}")
                    return result
                if not candidates:
                    result.update(status="invalid", error="API returned no candidates")
                    return result
                candidate = candidates[0]
                reason = candidate.get("finishReason")
                result["finish_reason"] = reason
                result["safety_ratings"] = candidate.get("safetyRatings", [])
                result["raw_text"] = "".join(part.get("text", "") for part in
                                               candidate.get("content", {}).get("parts", [])
                                               if not part.get("thought", False))
                if reason in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY"}:
                    result.update(status="blocked", error=f"Candidate blocked: {reason}")
                else:
                    result.update(status="ok", error=None)
                return result
            except urllib.error.HTTPError as exc:
                status = exc.code
                retry_delay = _retry_after(exc.headers)
                exc.close()
                result["attempt_log"].append({"attempt": attempt, "http_status": status})
                result["error"] = f"Gemini HTTP {status}"
                retryable = status in {408, 429, 500, 502, 503, 504}
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                result["attempt_log"].append({"attempt": attempt, "transport_error": type(exc).__name__})
                result["error"] = f"Gemini transport error: {type(exc).__name__}"
                retry_delay, retryable = 0.0, True
            except (ValueError, UnicodeError, TypeError, AttributeError) as exc:
                result["error"] = f"Malformed Gemini response: {type(exc).__name__}"
                return result
            if not retryable or attempt == self.max_attempts:
                return result
            backoff = max(retry_delay, min(60.0, 2.0 ** (attempt - 1)) + random.uniform(0, 0.5))
            # Every credential observes the same cooldown. Keys do not bypass project quotas.
            self._quota.penalize(backoff)
        return result


def list_models():
    key = load_keys()[0]
    models = []
    token = None
    import urllib.parse
    while True:
        suffix = "?pageSize=1000" + ("&pageToken=" + urllib.parse.quote(token, safe="") if token else "")
        request = urllib.request.Request(API_BASE + "/models" + suffix,
                                         headers={"x-goog-api-key": key})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                page = strict_json(response.read().decode())
        except urllib.error.HTTPError as exc:
            code = exc.code
            exc.close()
            raise ValueError(f"Gemini model listing failed: HTTP {code}") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ValueError("Gemini model listing failed: network/timeout error") from None
        for model in page.get("models", []):
            if "generateContent" in model.get("supportedGenerationMethods", []):
                models.append({key: model.get(key) for key in
                               ("name", "version", "displayName", "inputTokenLimit", "outputTokenLimit")})
        token = page.get("nextPageToken")
        if not token:
            return {"models": models}
