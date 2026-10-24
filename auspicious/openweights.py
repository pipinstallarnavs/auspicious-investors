"""Optional local inference, correlational probes and steering sweeps; no mediation claims.

Only explicitly selected Hugging Face decoder models are loaded. Residuals are
captured at decoder-block output, before the model's final normalization, at the
final prompt token. Steering adds a direction at that same block; the token
positions that receive it (``steering_positions``) are an explicit, recorded
setting: ``all`` (every forward token, including cached decoding),
``prompt_last`` (only the final prompt token during prefill, which matches the
extraction site) or ``generated`` (only decode steps after prefill).

Probe accuracy is decodability of prompt wording/condition from activations. It
is reported next to a character n-gram text baseline, held-out
festival/language/template/paraphrase transfer and a random-direction null so
that recognition of surface text is not mistaken for a representation. None of
these outputs establish mediation of behavior; behavioral steering effects must
be compared against the zero baseline, random directions and a capability check.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import threading
import zlib


_FORMAT = 1
_REPRESENTATION = "decoder_block_output_prompt_final_token"
_STEERING_POSITIONS = ("all", "prompt_last", "generated")
_PROBE_CONDITIONS = ("event", "neutral")
_TRANSFER_FIELDS = ("festival", "language", "template_id", "paraphrase_id", "cue")
_TEXT_DIM = 2 ** 14
_TEXT_NGRAMS = (3, 4, 5)
_NULL_DIRECTIONS = 200
_NULL_SEED = 4242
_CAPABILITY_LANGUAGES = ("en", "hi", "ta", "gu", "bn")
_text_cache = {}


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ValueError("Install the openweights extra (numpy, torch, transformers).") from exc
    return np


def _load_runtime(model, revision, device):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ValueError("Install the openweights extra (numpy, torch, transformers).") from exc
    try:
        tokenizer = AutoTokenizer.from_pretrained(model, revision=revision, trust_remote_code=False)
        network = AutoModelForCausalLM.from_pretrained(
            model, revision=revision, trust_remote_code=False
        ).to(device)
        network.eval()
    except Exception as exc:
        raise ValueError(f"Cannot load explicitly selected model {model!r}: {exc}") from exc
    return torch, tokenizer, network


def _decoder_layers(network):
    layers = getattr(getattr(network, "model", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("Unsupported architecture: expected decoder blocks at model.model.layers.")
    if getattr(network.config, "is_encoder_decoder", False):
        raise ValueError("Unsupported architecture: encoder-decoder models are not supported.")
    return layers


def _layer_index(layer, count):
    if isinstance(layer, bool) or not isinstance(layer, int) or not -count <= layer < count:
        raise ValueError(f"layer must be an integer in [{-count}, {count - 1}].")
    return layer % count


def _width(network):
    width = getattr(network.config, "hidden_size", None)
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("Unsupported architecture: config.hidden_size is required.")
    return width


def _hidden(output):
    value = output[0] if isinstance(output, tuple) else output
    if not hasattr(value, "ndim") or value.ndim != 3:
        raise ValueError("Unsupported decoder output: expected [batch, tokens, hidden] tensor.")
    return value


def _inputs(tokenizer, prompt, device):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Prompt must be nonempty text.")
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False,
                            return_token_type_ids=False, truncation=False)
    else:
        encoded = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False, truncation=False)
    return {key: value.to(device) for key, value in encoded.items()}


def _local_model_digest(model):
    root = Path(model)
    if not root.is_dir():
        return None
    digest = hashlib.sha256()
    try:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                name = path.relative_to(root).as_posix().encode("utf-8")
                digest.update(len(name).to_bytes(8, "big"))
                digest.update(name)
                digest.update(bytes.fromhex(_artifact_digest(path)))
    except OSError as exc:
        raise ValueError(f"Cannot fingerprint local model/tokenizer files: {exc}") from exc
    return digest.hexdigest()


def _provenance(network, model, revision, layer, *, local_model_sha256=None, tokenizer=None):
    from .protocol import runtime_identity
    return {
        "format_version": _FORMAT, "model": model, "revision": revision,
        "resolved_revision": getattr(network.config, "_commit_hash", None),
        "tokenizer_resolved_revision": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "runtime": runtime_identity(),
        "local_model_sha256": local_model_sha256 if local_model_sha256 is not None else _local_model_digest(model),
        "layer": layer, "width": _width(network), "representation": _REPRESENTATION,
        "prompt_format": "tokenizer_chat_template_if_available_else_plain",
        "trust_remote_code": False,
    }


def _atomic_npz(path, **arrays):
    np = _numpy()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, target)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _json_array(value):
    return _numpy().asarray(json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True))


def _artifact_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class _ResidualSteering:
    """Forward hook adding ``strength * direction`` at selected token positions.

    The hook is stateful: one instance serves exactly one generation, and its
    first call is the prefill forward (all prompt tokens at once). Later calls
    are cached decode steps. ``None`` is returned (output unchanged) whenever a
    call is not selected, so an unselected call is bitwise the unsteered model.
    """

    def __init__(self, direction, strength, positions="all"):
        if positions not in _STEERING_POSITIONS:
            raise ValueError(f"steering_positions must be one of {_STEERING_POSITIONS}.")
        self.direction = direction * strength
        self.positions = positions
        self.calls = 0

    def __call__(self, module, args, output):
        hidden = _hidden(output)
        if hidden.shape[-1] != self.direction.shape[0]:
            raise ValueError("Steering direction width does not match decoder output.")
        prefill = self.calls == 0
        self.calls += 1
        shift = self.direction.to(device=hidden.device, dtype=hidden.dtype)
        if self.positions == "all":
            shifted = hidden + shift
        elif self.positions == "prompt_last":
            if not prefill:
                return None
            shifted = hidden.clone()
            shifted[:, -1, :] += shift
        else:
            if prefill:
                return None
            shifted = hidden + shift
        return (shifted,) + output[1:] if isinstance(output, tuple) else shifted


def _read_steering(path, provenance, random_direction_seed=None, *, dose="raw_coefficient"):
    np = _numpy()
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            direction = np.asarray(data["direction"], dtype=np.float64).copy()
            scale = np.asarray(data["scaler_scale"], dtype=np.float64) if dose == "activation_std" else None
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid steering artifact: {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("kind") != "ridge_condition_probe":
        raise ValueError("Steering requires a ridge_condition_probe artifact.")
    for key in ("format_version", "model", "revision", "resolved_revision", "tokenizer_resolved_revision", "local_model_sha256", "layer", "width", "representation", "prompt_format"):
        if metadata.get(key) != provenance.get(key):
            raise ValueError(f"Steering provenance mismatch for {key}: {metadata.get(key)!r} != {provenance.get(key)!r}.")
    if direction.shape != (provenance["width"],) or not np.isfinite(direction).all():
        raise ValueError("Steering direction has invalid width or nonfinite values.")
    if dose not in ("raw_coefficient", "activation_std"):
        raise ValueError("Unknown steering dose calibration")
    if dose == "activation_std":
        if scale.shape != direction.shape or not np.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("Calibrated steering requires finite positive train activation scales")
        # A unit displacement in diagonal-standardized training activation space.
        # This is not a full covariance/Mahalanobis or downstream-disruption match.
        direction = direction * scale
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Steering direction must have a finite nonzero norm.")
    if random_direction_seed is not None:
        random = np.random.default_rng(random_direction_seed).normal(size=direction.shape)
        direction = random * (norm / np.linalg.norm(random))
    if dose == "activation_std":
        direction = direction / norm * scale
    return direction


class OpenWeightProvider:
    """Lazy, serialized inference; random control preserves the learned L2 norm.

    ``strength`` multiplies the raw-space classifier coefficient, not a unit
    vector. Positive strength points toward event rather than neutral prompts;
    this does not assume a direction of financial behavior or identify mediation.
    ``steering_positions`` selects where the vector is added (see module
    docstring) and is part of the recorded identity, so runs with different
    positions never share a fingerprint or analysis stratum. Zero strength
    registers no hook and is exactly the unsteered model.
    """

    def __init__(self, model, revision=None, device="cpu", temperature=0.0,
                 max_output_tokens=512, seed=42, steering_path=None, strength=0.0,
                 layer=None, *, random_direction_seed=None, steering_positions="all",
                 steering_dose="raw_coefficient", _shared=None):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("An explicit model name or local model directory is required.")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens < 1:
            raise ValueError("max_output_tokens must be a positive integer.")
        if not math.isfinite(strength):
            raise ValueError("strength must be finite.")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        if random_direction_seed is not None and (isinstance(random_direction_seed, bool) or not isinstance(random_direction_seed, int) or random_direction_seed < 0):
            raise ValueError("random_direction_seed must be a nonnegative integer.")
        if steering_positions not in _STEERING_POSITIONS:
            raise ValueError(f"steering_positions must be one of {_STEERING_POSITIONS}.")
        if steering_path is None and (strength != 0 or random_direction_seed is not None):
            raise ValueError("Nonzero steering and random controls require steering_path.")
        self.model_name, self.revision, self.device = model, revision, device
        self.temperature, self.max_output_tokens, self.seed = temperature, max_output_tokens, seed
        self.steering_path, self.strength, self.layer = steering_path, strength, layer
        self.random_direction_seed = random_direction_seed
        self.steering_positions = steering_positions
        if steering_dose not in ("raw_coefficient", "activation_std"):
            raise ValueError("Unknown steering dose calibration")
        self.steering_dose = steering_dose
        # _shared lets steering_grid reuse one loaded model and one local-file digest.
        self._preloaded = _shared["runtime"] if _shared else None
        self._runtime = None
        self._direction = None
        self._lock = threading.Lock()
        self._local_model_sha256 = _shared["local_model_sha256"] if _shared else _local_model_digest(model)
        try:
            digest = _artifact_digest(steering_path) if steering_path else None
        except OSError as exc:
            raise ValueError(f"Cannot read steering artifact: {exc}") from exc
        self.identity = {
            "provider": "openweights", "requested_model": model, "revision": revision,
            "device": device, "steering_sha256": digest,
            "local_model_sha256": self._local_model_sha256,
            "generation": {"temperature": temperature, "max_output_tokens": max_output_tokens,
                           "seed": seed, "strength": strength, "layer": layer,
                           "random_direction_seed": random_direction_seed,
                           "steering_sha256": digest,
                           "steering_dose": steering_dose,
                           "steering_positions": steering_positions},
        }

    def _ensure_loaded(self):
        if self._runtime is not None:
            return
        runtime = self._preloaded or _load_runtime(self.model_name, self.revision, self.device)
        torch, tokenizer, network = runtime
        layers = _decoder_layers(network)
        selected = self.layer
        if self.steering_path:
            if _artifact_digest(self.steering_path) != self.identity["steering_sha256"]:
                raise ValueError("Steering artifact changed after provider initialization.")
            if selected is None:
                np = _numpy()
                try:
                    with np.load(self.steering_path, allow_pickle=False) as data:
                        selected = json.loads(str(data["metadata"].item()))["layer"]
                except (OSError, KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid steering artifact: {exc}") from exc
        selected = _layer_index(-1 if selected is None else selected, len(layers))
        provenance = _provenance(network, self.model_name, self.revision, selected,
                                 local_model_sha256=self._local_model_sha256, tokenizer=tokenizer)
        if self.steering_path:
            direction = _read_steering(self.steering_path, provenance, self.random_direction_seed, dose=self.steering_dose)
            self._direction = torch.as_tensor(
                direction, device=self.device, dtype=next(network.parameters()).dtype
            )
        self._selected_layer = selected
        self._runtime = runtime

    def prepare(self):
        """Resolve checkpoint/hook compatibility before journaling or scheduling trials."""
        with self._lock:
            self._ensure_loaded()
            network = self._runtime[2]
            self.identity["resolved_revision"] = getattr(network.config, "_commit_hash", None)
            self.identity["tokenizer_resolved_revision"] = getattr(self._runtime[1], "init_kwargs", {}).get("_commit_hash")
            self.identity["generation"]["resolved_layer"] = self._selected_layer

    def _seed_for(self, key):
        material = f"{self.seed}:{key}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)

    def generate(self, prompt):
        return self._generate(prompt, self.seed)

    def generate_text(self, prompt, key=None):
        """Unvalidated completion (for capability items); ``key`` derives the sampling seed."""
        return self._generate(prompt, self.seed if key is None else self._seed_for(key), validate=False)

    def generate_trial(self, trial):
        from .core import normalize_result
        return normalize_result(self._generate(trial["prompt"], self._seed_for(trial["trial_id"])), trial=trial)

    def _generate(self, prompt, sampling_seed, validate=True):
        # A provider may be shared by run_trials worker threads. Temporary hooks
        # and sampling RNG state must not leak between concurrent requests.
        with self._lock:
            self._ensure_loaded()
            torch, tokenizer, network = self._runtime
            inputs = _inputs(tokenizer, prompt, self.device)
            count = int(inputs["input_ids"].shape[-1])
            if count < 1:
                raise ValueError("Tokenizer produced an empty prompt.")
            kwargs = {"max_new_tokens": self.max_output_tokens,
                      "do_sample": self.temperature > 0, "use_cache": True}
            if self.temperature > 0:
                kwargs["temperature"] = self.temperature
            pad = getattr(tokenizer, "pad_token_id", None)
            eos = getattr(tokenizer, "eos_token_id", None)
            if pad is not None or eos is not None:
                kwargs["pad_token_id"] = pad if pad is not None else eos
            hook = None
            try:
                if self._direction is not None and self.strength != 0:
                    hook = _decoder_layers(network)[self._selected_layer].register_forward_hook(
                        _ResidualSteering(self._direction, self.strength, self.steering_positions)
                    )
                cuda_devices = []
                target_device = torch.device(self.device)
                if target_device.type == "cuda":
                    cuda_devices = [target_device.index if target_device.index is not None else torch.cuda.current_device()]
                with torch.inference_mode(), torch.random.fork_rng(devices=cuda_devices):
                    torch.random.default_generator.manual_seed(sampling_seed)
                    if cuda_devices:
                        with torch.cuda.device(cuda_devices[0]):
                            torch.cuda.manual_seed(sampling_seed)
                    generated = network.generate(**inputs, **kwargs)
                continuation = generated[0, count:]
                raw = tokenizer.decode(continuation, skip_special_tokens=True)
            finally:
                if hook is not None:
                    hook.remove()
            result = {
                "status": "ok", "raw_text": raw, "payload": None,
                "provider": "openweights", "requested_model": self.model_name,
                "model_version": getattr(network.config, "_commit_hash", None) or self._local_model_sha256 or self.revision,
                "generation": dict(self.identity["generation"], resolved_layer=self._selected_layer),
                "usage": {"input_tokens": count, "output_tokens": int(continuation.shape[-1])},
                "attempts": 1, "error": None,
                "sampling_seed": sampling_seed,
            }
            if not validate:
                return result
            from .core import normalize_result
            return normalize_result(result)


def _trials_digest(trials):
    return hashlib.sha256(
        json.dumps(trials, ensure_ascii=False, allow_nan=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _probe_trials(trials):
    """Probe label is event vs neutral; control-condition prompts are not part of it."""
    return [row for row in trials if row["condition"] in _PROBE_CONDITIONS]


def _capture_activations(torch, tokenizer, network, device, trials, layer_indices, width):
    """Final-prompt-token residuals of several blocks in one forward pass per prompt."""
    np = _numpy()
    layers = _decoder_layers(network)
    features = {index: np.empty((len(trials), width), dtype=np.float32) for index in layer_indices}
    captured = {index: [] for index in layer_indices}

    def capture_for(index):
        def capture(module, args, output):
            hidden = _hidden(output)
            if hidden.shape[0] != 1 or hidden.shape[-1] != width:
                raise ValueError("Unexpected residual shape during activation extraction.")
            captured[index].append(hidden[0, -1].detach().float().cpu().numpy().copy())
        return capture

    hooks = []
    try:
        for index in layer_indices:
            hooks.append(layers[index].register_forward_hook(capture_for(index)))
        with torch.inference_mode():
            for row, trial in enumerate(trials):
                for values in captured.values():
                    values.clear()
                inputs = _inputs(tokenizer, trial["prompt"], device)
                network(**inputs, use_cache=False)
                for index in layer_indices:
                    if len(captured[index]) != 1:
                        raise ValueError("Unsupported architecture: selected block must run exactly once per prompt.")
                    features[index][row] = captured[index][0]
    finally:
        for hook in hooks:
            hook.remove()
    for matrix in features.values():
        if not np.isfinite(matrix).all():
            raise ValueError("Model produced nonfinite residual activations.")
    return features


def _save_activations(output_path, trials, features, provenance, skipped):
    np = _numpy()
    metadata = dict(provenance, kind="prompt_activations", n_trials=len(trials),
                    label="calendar salience:condition:event=1,neutral=0",
                    event_valences=sorted({row["event_valence"] for row in trials}),
                    conditions=list(_PROBE_CONDITIONS), n_control_trials_excluded=skipped,
                    trials_sha256=_trials_digest(trials))
    _atomic_npz(output_path, features=features,
                labels=np.asarray([int(row["condition"] == "event") for row in trials], dtype=np.int8),
                trial_ids=np.asarray([row["trial_id"] for row in trials]),
                cluster_ids=np.asarray([row["cluster_id"] for row in trials]),
                splits=np.asarray([row["split"] for row in trials]),
                trials=_json_array(trials), metadata=_json_array(metadata))
    return metadata


def extract_activations(trials_path, output_path, *, model, revision=None, layer=-1, device="cpu"):
    """Capture prompt-only residuals without generating or priming decision runs.

    Only ``event`` and ``neutral`` trials are extracted (the probe label); control
    conditions in the corpus are excluded and counted in the metadata. The trial
    rows are stored in the artifact so the text baseline sees the same prompts.
    """
    from .core import read_jsonl, validate_trials
    np = _numpy()
    all_trials = read_jsonl(trials_path)
    validate_trials(all_trials)
    trials = _probe_trials(all_trials)
    if not trials:
        raise ValueError("Activation extraction requires at least one event/neutral trial.")
    torch, tokenizer, network = _load_runtime(model, revision, device)
    layers = _decoder_layers(network)
    selected = _layer_index(layer, len(layers))
    provenance = _provenance(network, model, revision, selected, tokenizer=tokenizer)
    features = _capture_activations(torch, tokenizer, network, device, trials, [selected],
                                    provenance["width"])[selected]
    metadata = _save_activations(output_path, trials, features, provenance, len(all_trials) - len(trials))
    return dict(metadata, output=str(output_path))


def _ridge_fit(np, x, labels, ridge):
    target = 2.0 * labels - 1.0
    mean_x, mean_y = x.mean(axis=0), float(target.mean())
    centered_x, centered_y = x - mean_x, target - mean_y
    n, width = x.shape
    if width <= n:
        gram = centered_x.T @ centered_x
        gram.flat[::width + 1] += ridge
        coefficient = np.linalg.solve(gram, centered_x.T @ centered_y)
    else:
        gram = centered_x @ centered_x.T
        gram.flat[::n + 1] += ridge
        coefficient = centered_x.T @ np.linalg.solve(gram, centered_y)
    return coefficient, float(mean_y - mean_x @ coefficient)


def _decision_scores(np, decision, labels, clusters):
    predicted = (decision >= 0).astype(np.int8)
    correct = predicted == labels
    cluster_accuracy = [float(correct[clusters == key].mean()) for key in np.unique(clusters)]
    balanced = [float(correct[labels == value].mean()) for value in (0, 1) if (labels == value).any()]
    loo = [float(np.mean(cluster_accuracy[:i] + cluster_accuracy[i + 1:]))
           for i in range(len(cluster_accuracy))] if len(cluster_accuracy) > 1 else []
    return {"n": int(len(labels)), "n_clusters": int(len(cluster_accuracy)),
            "accuracy": float(correct.mean()), "balanced_accuracy": float(np.mean(balanced)),
            "cluster_macro_accuracy": float(np.mean(cluster_accuracy)),
            "per_cluster_accuracy": {str(key): float(correct[clusters == key].mean()) for key in np.unique(clusters)},
            "leave_one_cluster_out_macro_range": [min(loo), max(loo)] if loo else None,
            "uncertainty_note": "Descriptive scenario sensitivity; no population confidence interval."}


def _scores(np, x, labels, clusters, coefficient, intercept):
    return _decision_scores(np, x @ coefficient + intercept, labels, clusters)


def _standardized_ridge(np, x, labels, train, ridge):
    """Ridge on ``x[train]`` with scaler statistics from the training rows only."""
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale == 0] = 1.0
    coefficient, intercept = _ridge_fit(np, (x[train] - mean) / scale, labels[train], ridge)
    return mean, scale, coefficient, intercept


# --- text baseline -------------------------------------------------------------

def _text_vector(np, text):
    """Signed-hashed log-count character 3-5-gram vector, L2-normalized (no fitted state)."""
    vector = np.zeros(_TEXT_DIM, dtype=np.float32)
    for size in _TEXT_NGRAMS:
        counts = {}
        for start in range(len(text) - size + 1):
            gram = text[start:start + size]
            counts[gram] = counts.get(gram, 0) + 1
        for gram, count in counts.items():
            hashed = zlib.crc32(gram.encode("utf-8"))
            sign = -1.0 if (hashed >> 16) & 1 else 1.0
            vector[hashed & (_TEXT_DIM - 1)] += sign * math.log1p(count)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector /= norm
    return vector


def _text_kernel(np, prompts):
    """Linear kernel of the text features over unique prompts, plus a row->prompt index.

    Neutral prompt text recurs across festivals, so only distinct prompts are
    vectorized. The most recent kernel is cached (layers of one corpus share it).
    """
    position = {}
    for prompt in prompts:
        position.setdefault(prompt, len(position))
    key = hashlib.sha256("\0".join(position).encode("utf-8")).hexdigest()
    if key not in _text_cache:
        matrix = np.stack([_text_vector(np, prompt) for prompt in position])
        _text_cache.clear()
        _text_cache[key] = (matrix @ matrix.T).astype(np.float64)
    inverse = np.asarray([position[prompt] for prompt in prompts], dtype=np.int64)
    return _text_cache[key], inverse


def _kernel_decision(np, kernel, inverse, train, evaluate, labels, ridge):
    """Dual-form ridge identical to ``_ridge_fit`` on centered features (train-only centering)."""
    rows, other = inverse[train], inverse[evaluate]
    gram = kernel[np.ix_(rows, rows)]
    row_mean = gram.mean(axis=1)
    grand = float(row_mean.mean())
    centered = gram - row_mean[:, None] - row_mean[None, :] + grand
    target = 2.0 * labels[train] - 1.0
    mean_y = float(target.mean())
    centered.flat[::len(train) + 1] += ridge
    alpha = np.linalg.solve(centered, target - mean_y)
    cross = kernel[np.ix_(other, rows)]
    cross = cross - cross.mean(axis=1)[:, None] - row_mean[None, :] + grand
    return cross @ alpha + mean_y


def _tune_text_ridge(np, text, labels, clusters, train, dev):
    candidates = (0.01, 0.1, 1.0, 10.0, 100.0)
    if not len(dev) or len(np.unique(labels[dev])) != 2:
        return 1.0, {"status": "fixed_fallback_no_eligible_dev", "candidates": list(candidates)}
    scores = {value: _decision_scores(np, _kernel_decision(np, *text, train, dev, labels, value),
                                     labels[dev], clusters[dev])["cluster_macro_accuracy"] for value in candidates}
    chosen = max(candidates, key=lambda value: (scores[value], -value))
    return chosen, {"status": "dev_cluster_macro_accuracy", "scores": scores, "tie_break": "smallest ridge"}


def _text_baseline(np, trials, labels, clusters, splits, ridge, text, evaluation_split="test"):
    kernel, inverse = text
    train = np.flatnonzero(splits == "train")
    ridge, selection = _tune_text_ridge(np, text, labels, clusters, train, np.flatnonzero(splits == "dev"))
    majority = int(labels[train].mean() >= 0.5)
    reports = {}
    for name in ("train", "dev", "test"):
        if name == "test" and evaluation_split != "test":
            continue
        rows = np.flatnonzero(splits == name)
        decision = _kernel_decision(np, kernel, inverse, train, rows, labels, ridge)
        reports[name] = _decision_scores(np, decision, labels[rows], clusters[rows])
        reports[name]["majority_baseline_accuracy"] = float((labels[rows] == majority).mean())
    return {
        "features": f"signed hashed log-count character {_TEXT_NGRAMS[0]}-{_TEXT_NGRAMS[-1]}-grams, "
                    f"dim {_TEXT_DIM}, L2-normalized rows; trial prompt text only",
        "ridge": float(ridge), "ridge_note": "text penalty tuned independently of activation penalty on dev only",
        "selection": selection,
        "splits": reports,
    }


def _validated_trials(np, text, ids, labels, clusters, splits):
    try:
        trials = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"Invalid stored trials: {exc}") from exc
    if not isinstance(trials, list) or len(trials) != len(ids):
        raise ValueError("Stored trials must match activation rows.")
    for row, trial in enumerate(trials):
        if not isinstance(trial, dict) or not isinstance(trial.get("prompt"), str):
            raise ValueError("Stored trials must be trial objects with prompts.")
        if trial.get("prompt_sha256") != hashlib.sha256(trial["prompt"].encode()).hexdigest():
            raise ValueError("Stored trial prompt hash mismatch")
        if (trial.get("trial_id") != str(ids[row]) or trial.get("cluster_id") != str(clusters[row])
                or trial.get("split") != str(splits[row])
                or int(trial.get("condition") == "event") != int(labels[row])):
            raise ValueError("Stored trials disagree with activation labels/splits/clusters.")
    return trials


# --- transfer and random null --------------------------------------------------

def _macro(rows, key):
    return float(sum(row[key] for row in rows) / len(rows))


def _transfer(np, x, labels, clusters, splits, trials, ridge, text, evaluation_split="test"):
    kernel, inverse = text
    report = {
        "design": "leave-one-value-out: probe and text baseline are fit on train-split clusters "
                  "whose value differs from the held-out value, scored on test-split clusters of "
                  "the held-out value; train-only scaling; no tuning",
    }
    for field in _TRANSFER_FIELDS:
        if any(field not in trial for trial in trials):
            report[field] = {"skipped": f"trials do not carry {field}"}
            continue
        values = np.asarray([str(trial[field]) for trial in trials])
        distinct = sorted(set(values.tolist()))
        if len(distinct) < 2:
            report[field] = {"skipped": f"only {len(distinct)} distinct {field} value present; need >=2"}
            continue
        held_out, evaluated = {}, []
        for value in distinct:
            train = np.flatnonzero((splits == "train") & (values != value))
            test = np.flatnonzero((splits == evaluation_split) & (values == value))
            if not len(test) or len(np.unique(labels[test])) != 2:
                held_out[value] = {"skipped": "held-out test split lacks both event and neutral trials"}
                continue
            if not len(train) or len(np.unique(labels[train])) != 2:
                held_out[value] = {"skipped": "remaining train split lacks both event and neutral trials"}
                continue
            mean, scale, coefficient, intercept = _standardized_ridge(np, x, labels, train, ridge)
            probe = _decision_scores(np, ((x[test] - mean) / scale) @ coefficient + intercept,
                                     labels[test], clusters[test])
            text_ridge, _selection = _tune_text_ridge(np, text, labels, clusters, train,
                                                     np.flatnonzero((splits == "dev") & (values != value)))
            baseline = _decision_scores(
                np, _kernel_decision(np, kernel, inverse, train, test, labels, text_ridge),
                labels[test], clusters[test])
            row = {"n": probe["n"], "n_clusters": probe["n_clusters"], "n_train": int(len(train)),
                   "accuracy": probe["accuracy"], "balanced_accuracy": probe["balanced_accuracy"],
                   "cluster_macro_accuracy": probe["cluster_macro_accuracy"],
                   "leave_one_cluster_out_macro_range": probe["leave_one_cluster_out_macro_range"],
                   "text_ridge": text_ridge,
                   "text_baseline_accuracy": baseline["accuracy"],
                   "text_baseline_balanced_accuracy": baseline["balanced_accuracy"],
                   "text_baseline_cluster_macro_accuracy": baseline["cluster_macro_accuracy"]}
            held_out[value] = row
            evaluated.append(row)
        entry = {"held_out": held_out, "n_values_evaluated": len(evaluated)}
        if evaluated:
            entry.update(macro_accuracy=_macro(evaluated, "accuracy"),
                         macro_text_baseline_accuracy=_macro(evaluated, "text_baseline_accuracy"),
                         macro_difference=_macro(evaluated, "accuracy") - _macro(evaluated, "text_baseline_accuracy"))
        else:
            entry["skipped"] = "no held-out value had both classes in train and test"
        report[field] = entry
    if "language" in report:
        report["language"]["note"] = "cross-language transfer"
    report["evaluation_split"] = evaluation_split
    report["design"] = "Train on other values; evaluate held-out values on " + evaluation_split + "; descriptive condition decodability. Text ridge selected on other-value dev clusters."
    report["limitations"] = "Cross-cue transfer does not certify lexical disjointness or polarity counterfactual validation."
    return report


def _random_null(np, standardized, labels, clusters, splits, coefficient, probe_accuracy, evaluation_split="test"):
    """Accuracy of norm-matched random directions; a diagnostic, not a calibrated test."""
    train, test = splits == "train", splits == evaluation_split
    rng = np.random.default_rng(_NULL_SEED)
    directions = rng.normal(size=(_NULL_DIRECTIONS, standardized.shape[1]))
    directions *= float(np.linalg.norm(coefficient)) / np.linalg.norm(directions, axis=1, keepdims=True)
    # Training rows are standardized to mean zero, so the train-only ridge intercept is
    # the mean of the +/-1 training targets for every direction.
    intercept = float((2.0 * labels[train] - 1.0).mean())
    decision = standardized[test] @ directions.T + intercept
    accuracy = ((decision >= 0) == labels[test][:, None].astype(bool)).mean(axis=0)
    return {
        "n_directions": _NULL_DIRECTIONS, "seed": _NULL_SEED, "split": evaluation_split,
        "intercept": "fit on train only (mean +/-1 training target)",
        "accuracy_mean": float(accuracy.mean()),
        "accuracy_p5": float(np.percentile(accuracy, 5)),
        "accuracy_p95": float(np.percentile(accuracy, 95)),
        "fraction_ge_probe_accuracy": float((accuracy >= probe_accuracy - 1e-12).mean()),
        "probe_test_accuracy": float(probe_accuracy),
        "interpretation": "Diagnostic spread of chance-level directions with the same norm; not a calibrated "
                          "significance test (directions are isotropic, the test split has few clusters).",
    }


def _probe_vs_text(probe_report, text_report, evaluation_split="test"):
    probe, text = probe_report["accuracy"], text_report["accuracy"]
    difference = float(probe - text)
    interpretation = ("Both scores describe condition decodability. The condition is explicit in the prompt; "
                      "a ceiling text score cannot be exceeded. Neither the difference nor random directions "
                      "establish information beyond wording, a general auspiciousness concept, or mediation.")
    result = {"probe_accuracy": float(probe), "text_baseline_accuracy": float(text),
              "evaluation_split": evaluation_split, "difference": difference,
              "same_splits": True, "interpretation": interpretation}
    if evaluation_split == "test":
        result.update(probe_test_accuracy=float(probe), text_baseline_test_accuracy=float(text))
    return result


def _training_cluster_sensitivity(np, x, labels, clusters, splits, ridge, evaluation_split):
    rows = []
    evaluate = splits == evaluation_split
    for cluster in np.unique(clusters[splits == "train"]):
        train = np.flatnonzero((splits == "train") & (clusters != cluster))
        if not len(train) or len(np.unique(labels[train])) < 2:
            rows.append({"omitted_cluster": str(cluster), "skipped": "remaining train lacks both classes"})
            continue
        mean, scale, coef, intercept = _standardized_ridge(np, x, labels, train, ridge)
        row = _scores(np, (x[evaluate] - mean) / scale, labels[evaluate], clusters[evaluate], coef, intercept)
        rows.append({"omitted_cluster": str(cluster), **row})
    return {"evaluation_split": evaluation_split, "refits": rows,
            "interpretation": "Descriptive leave-one-train-scenario-out stability; no population uncertainty guarantee."}


def fit_probe(activations_path, output_path, *, ridge=1.0, evaluation_split="test"):
    """Fit event-vs-neutral ridge classifier with strictly train-only scaling.

    Activation ridge is fixed; the text comparator selects its penalty on dev.
    Test scores are omitted during dev selection. A fixed-seed cluster-level label flip
    and training-majority classifier are negative/baseline controls, not a
    calibrated permutation significance test. The report also contains a
    character n-gram text baseline on the same cluster splits (``text_baseline``,
    ``probe_vs_text_baseline``), leave-one-value-out transfer over festival,
    language, template and paraphrase (``transfer``) and a random-direction
    null (``random_direction_null``). A fit accepts one event valence.
    Decodability does not establish mediation of financial decisions.
    """
    np = _numpy()
    if evaluation_split not in ("dev", "test"):
        raise ValueError("Probe evaluation_split must be dev or test")
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and strictly positive.")
    try:
        with np.load(activations_path, allow_pickle=False) as data:
            x = np.asarray(data["features"], dtype=np.float64)
            labels = np.asarray(data["labels"])
            ids = np.asarray(data["trial_ids"])
            clusters, splits = np.asarray(data["cluster_ids"]), np.asarray(data["splits"])
            metadata = json.loads(str(data["metadata"].item()))
            trials_text = str(data["trials"].item())
    except (OSError, KeyError, ValueError, TypeError) as exc:
        raise ValueError(f"Invalid activation artifact (stored trials are required for the text baseline): {exc}") from exc
    if not isinstance(metadata, dict) or metadata.get("kind") != "prompt_activations" or metadata.get("format_version") != _FORMAT or metadata.get("representation") != _REPRESENTATION:
        raise ValueError("Expected supported prompt-only activation artifact.")
    valences = metadata.get("event_valences")
    if valences not in (["auspicious"], ["avoidance"]):
        raise ValueError("Fit one event_valence at a time: subset whole trial pairs to auspicious or avoidance before extraction.")
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0 or not np.isfinite(x).all():
        raise ValueError("Features must be a nonempty finite matrix.")
    if x.shape[1] != metadata.get("width"):
        raise ValueError("Activation metadata width does not match features.")
    if any(values.shape != (x.shape[0],) for values in (labels, ids, clusters, splits)):
        raise ValueError("Activation metadata arrays must match the number of rows.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate trial IDs in activation artifact.")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Probe labels must be event=1 or neutral=0.")
    if not np.isin(splits, ["train", "dev", "test"]).all():
        raise ValueError("Every activation must have train, dev, or test split.")
    if any(not str(value).strip() for value in clusters):
        raise ValueError("Cluster IDs cannot be empty.")
    owners = {}
    for cluster, split in zip(clusters.tolist(), splits.tolist()):
        if cluster in owners and owners[cluster] != split:
            raise ValueError(f"Cluster leakage: {cluster!r} occurs in multiple splits.")
        owners[cluster] = split
    masks = {name: splits == name for name in ("train", "dev", "test")}
    for name, mask in masks.items():
        if not mask.any() or len(np.unique(labels[mask])) != 2:
            raise ValueError(f"Split {name!r} must contain both event and neutral observations.")
    trials = _validated_trials(np, trials_text, ids, labels, clusters, splits)
    if metadata.get("trials_sha256") != _trials_digest(trials):
        raise ValueError("Embedded trial-content digest mismatch")
    train = masks["train"]
    mean, scale = x[train].mean(axis=0), x[train].std(axis=0)
    scale[scale == 0] = 1.0
    standardized = (x - mean) / scale
    coefficient, intercept = _ridge_fit(np, standardized[train], labels[train], ridge)
    # Permute whole clusters' paired condition coding, rather than pretending
    # translated/paraphrased repetitions are independent exchangeable samples.
    shuffled = labels[train].copy()
    rng = np.random.default_rng(42)
    train_clusters = clusters[train]
    flipped = 0
    for cluster in np.unique(train_clusters):
        if rng.integers(0, 2):
            mask = train_clusters == cluster
            shuffled[mask] = 1 - shuffled[mask]
            flipped += 1
    null_coefficient, null_intercept = _ridge_fit(np, standardized[train], shuffled, ridge)
    majority = int(labels[train].mean() >= 0.5)
    reports = {}
    for name, mask in masks.items():
        if name == "test" and evaluation_split != "test":
            continue
        reports[name] = _scores(np, standardized[mask], labels[mask], clusters[mask], coefficient, intercept)
        reports[name]["majority_baseline_accuracy"] = float((labels[mask] == majority).mean())
        reports[name]["cluster_label_flip_control"] = _scores(
            np, standardized[mask], labels[mask], clusters[mask], null_coefficient, null_intercept
        )
    direction = coefficient / scale
    raw_intercept = float(intercept - mean @ direction)
    if not np.isfinite(direction).all() or not math.isfinite(raw_intercept):
        raise ValueError("Probe fit produced nonfinite coefficients.")
    text = _text_kernel(np, [trial["prompt"] for trial in trials])
    text_report = _text_baseline(np, trials, labels, clusters, splits, ridge, text, evaluation_split)
    summary = {
        "ridge": float(ridge), "splits": reports, "control_seed": 42, "evaluation_split": evaluation_split,
        "event_valence": valences[0], "label": "calendar salience:event=1,neutral=0",
        "control_flipped_train_clusters": flipped, "train_only_scaling": True,
        "direction_l2_norm": float(np.linalg.norm(direction)),
        "interpretation": "Correlational condition decodability; not mediation or evidence of a universal mechanism.",
        "positive_direction": "event versus neutral calendar salience, not auspiciousness or necessarily increased investment",
        "extraction_position": "prompt_final_token",
        "steering_positions_note": "Positions are chosen on the provider (all, prompt_last, generated); the probe is learned at the final prompt token only.",
        "selection": "fixed ridge; no dev/test tuning",
        "text_baseline": text_report,
        "probe_vs_text_baseline": _probe_vs_text(reports[evaluation_split], text_report["splits"][evaluation_split], evaluation_split),
        "training_cluster_sensitivity": _training_cluster_sensitivity(np, x, labels, clusters, splits, ridge, evaluation_split),
        "transfer": _transfer(np, x, labels, clusters, splits, trials, ridge, text, evaluation_split),
        "random_direction_null": _random_null(np, standardized, labels, clusters, splits, coefficient,
                                              reports[evaluation_split]["accuracy"], evaluation_split),
    }
    provenance = {key: metadata.get(key) for key in (
        "format_version", "model", "revision", "resolved_revision", "tokenizer_resolved_revision", "local_model_sha256", "layer", "width", "runtime",
        "representation", "prompt_format", "trust_remote_code"
    )}
    artifact_metadata = dict(provenance, kind="ridge_condition_probe", report=summary,
                             activation_sha256=_artifact_digest(activations_path))
    _atomic_npz(output_path, direction=direction, intercept=np.asarray(raw_intercept),
                scaler_mean=mean, scaler_scale=scale, standardized_coefficient=coefficient,
                standardized_intercept=np.asarray(intercept), metadata=_json_array(artifact_metadata))
    return dict(summary, output=str(output_path), provenance=provenance)


# --- capability check ----------------------------------------------------------

_CAPABILITY_FORMULAS = {
    "percent_of": lambda p: p["amount"] * p["percent"] / 100,
    "compound_growth": lambda p: p["principal"] * (1 + p["rate_pct"] / 100) ** p["years"],
    "cash_remainder": lambda p: p["total"] * (100 - p["gold_pct"] - p["equity_pct"]) / 100,
    "return_difference": lambda p: p["principal"] * (p["rate_a_pct"] - p["rate_b_pct"]) / 100,
    "expected_value": lambda p: p["prob_pct"] / 100 * p["gain"] - (1 - p["prob_pct"] / 100) * p["loss"],
}


def _default_capability_path():
    return Path(__file__).resolve().parent.parent / "data" / "capability.json"


def _load_capability(items_path):
    from .core import strict_json
    path = Path(items_path) if items_path is not None else _default_capability_path()
    try:
        raw = path.read_bytes()
        data = strict_json(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read capability items {str(path)!r}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1 or not isinstance(data.get("languages"), dict) \
            or not isinstance(data.get("items"), list) or not data["items"]:
        raise ValueError("Capability file must have schema_version 1, languages and items.")
    if not str(data.get("status", "")).startswith("DRAFT"):
        raise ValueError("Capability data must be marked DRAFT until independently reviewed.")
    languages = list(data["languages"])
    ids = set()
    for item in data["items"]:
        if not isinstance(item, dict) or item.get("kind") not in _CAPABILITY_FORMULAS or not isinstance(item.get("params"), dict):
            raise ValueError("Capability item needs a known kind and params.")
        if item.get("id") in ids or not isinstance(item.get("id"), str):
            raise ValueError("Capability item ids must be unique strings.")
        ids.add(item["id"])
        try:
            expected = _CAPABILITY_FORMULAS[item["kind"]](item["params"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Capability item {item['id']} has invalid params: {exc}") from exc
        answer = item.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or abs(answer - expected) > 1e-6:
            raise ValueError(f"Capability item {item['id']}: stored answer {answer!r} != recomputed {expected!r}.")
    for language in languages:
        block = data["languages"][language]
        if not isinstance(block, dict) or not isinstance(block.get("instruction"), str) \
                or not {item["kind"] for item in data["items"]} <= set(block.get("templates", {})):
            raise ValueError(f"Capability language {language!r} lacks an instruction or templates.")
    return data, hashlib.sha256(raw).hexdigest(), str(path)


def _capability_prompt(data, language, item):
    block = data["languages"][language]
    try:
        question = block["templates"][item["kind"]].format(**item["params"])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(f"Capability template {language}/{item['kind']} does not match params: {exc}") from exc
    return f"{question}\n{block['instruction']}"


def _parse_answer(raw):
    from .core import strict_json
    try:
        value = strict_json(raw.strip())
    except (ValueError, TypeError, AttributeError):
        return None, "not strict JSON"
    if not isinstance(value, dict) or set(value) != {"answer"}:
        return None, "JSON object must have exactly the key 'answer'"
    answer = value["answer"]
    if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(answer):
        return None, "answer must be a finite number"
    return float(answer), None


def _answer_correct(answer, gold):
    difference = abs(answer - gold)
    return difference <= 0.01 or difference <= 0.001 * abs(gold)


def capability_check(output_path, provider, *, items_path=None, limit=None):
    """Arithmetic/financial-reasoning check with verifiable numeric answers.

    Runs every item in every language of the (DRAFT) capability file with the
    given provider, steering settings included, and scores strict JSON
    ``{"answer": number}`` replies: correct when within 0.01 absolute or 0.1%
    relative. Invalid replies count as incorrect and are reported as an invalid
    rate. ``limit`` keeps the first N items per language. Use it to see whether
    a steered or random-direction run disrupts general competence; it says
    nothing about calendar sensitivity.
    """
    from .core import atomic_text
    from .protocol import runtime_identity
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
        raise ValueError("limit must be a positive integer.")
    data, items_sha256, resolved_path = _load_capability(items_path)
    items = data["items"] if limit is None else data["items"][:limit]
    if hasattr(provider, "prepare"):
        provider.prepare()
    records = []
    for language in data["languages"]:
        for item in items:
            prompt = _capability_prompt(data, language, item)
            result = (provider.generate_text(prompt, key=f"capability:{language}:{item['id']}")
                      if hasattr(provider, "generate_text") else provider.generate(prompt))
            raw = result.get("raw_text", "") if result.get("status") not in ("blocked", "error") else ""
            answer, problem = _parse_answer(raw) if raw else (None, "no output")
            records.append({
                "language": language, "item_id": item["id"], "kind": item["kind"], "gold": item["answer"],
                "raw_text": raw, "parsed_answer": answer, "invalid_reason": problem,
                "valid": problem is None, "correct": problem is None and _answer_correct(answer, item["answer"]),
                "usage": result.get("usage", {}),
            })
    per_language = {}
    for language in data["languages"]:
        rows = [row for row in records if row["language"] == language]
        per_language[language] = {
            "n": len(rows), "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "invalid_rate": sum(not r["valid"] for r in rows) / len(rows),
            "accuracy_among_valid": (sum(r["correct"] for r in rows) / max(1, sum(r["valid"] for r in rows))
                                     if any(r["valid"] for r in rows) else None),
        }
    report = {
        "schema_version": 1, "kind": "capability_check",
        "items_status": data["status"], "items_path": resolved_path, "items_sha256": items_sha256,
        "n_items_per_language": len(items), "n_independent_authored_items": len(items), "limit": limit,
        "tolerance": "correct if |answer-gold| <= 0.01 or <= 0.1% of |gold|; strict JSON {\"answer\": number}",
        "provider_identity": json.loads(json.dumps(provider.identity)),
        "runtime": runtime_identity(),
        "overall": {"n": len(records), "accuracy": sum(r["correct"] for r in records) / len(records),
                    "invalid_rate": sum(not r["valid"] for r in records) / len(records)},
        "per_language": per_language, "records": records,
        "interpretation": "Performance on these draft arithmetic items. Translations are repeated measurements of the same items; no general reasoning or specificity claim.",
    }
    atomic_text(output_path, json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return dict(report, output=str(output_path))


# --- steering grid -------------------------------------------------------------

def _strength_name(value):
    value = float(value)
    return str(int(value)) if value.is_integer() else repr(value)


def _grid_sequence(name, values, convert):
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError(f"{name} must be a sequence.")
    result = [convert(value) for value in values]
    return result


def _int_value(name):
    def convert(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} entries must be integers.")
        return value
    return convert


def _strength_value(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("strengths must be finite numbers.")
    return value


def _artifact_metadata(path):
    np = _numpy()
    try:
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
    except (OSError, KeyError, ValueError, TypeError):
        return None
    return metadata if isinstance(metadata, dict) else None


def _read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _capability_report(output_path, provider, *, items_path=None, limit=None):
    """Reuse scores only for unchanged, currently valid capability materials and settings."""
    _data, items_sha256, resolved_path = _load_capability(items_path)
    from .protocol import runtime_identity
    current = _read_json_file(output_path)
    expected_identity = json.loads(json.dumps(provider.identity))
    if (current is not None and current.get("kind") == "capability_check"
            and current.get("provider_identity") == expected_identity
            and current.get("limit") == limit and current.get("items_sha256") == items_sha256
            and current.get("items_path") == resolved_path):
        if current.get("runtime") == runtime_identity():
            return current
    return capability_check(output_path, provider, items_path=resolved_path, limit=limit)


def freeze_steering_selection(grid_path, output_path, *, layer, position, strength):
    """Bind one completed dev cell before evaluating its held-out behavior."""
    from .core import atomic_text
    manifest = _read_json_file(grid_path)
    if not manifest or manifest.get("evaluation_split") != "dev" or not manifest.get("complete"):
        raise ValueError("Selection requires a completed dev grid")
    if not manifest.get("controls_complete"):
        raise ValueError("Selection requires random controls and capability checks")
    cells = [row for row in manifest["runs"] if row["kind"] == "learned" and row["layer"] == layer
             and row["position"] == position and row["strength"] == strength]
    if len(cells) != 1 or strength == 0:
        raise ValueError("Select exactly one nonzero learned dev cell")
    probe = manifest["probes"][str(layer)]
    payload = {"kind": "frozen_steering_selection", "schema_version": 1,
               "dev_grid_sha256": _artifact_digest(grid_path), "dev_grid_path": str(Path(grid_path).resolve()),
               "settings": manifest["settings"], "layer": layer, "position": position, "strength": strength,
               "probe_path": str(Path(probe["path"]).resolve()), "probe_sha256": probe["sha256"],
               "activation_path": str(Path(probe["activations"]).resolve()),
               "activation_sha256": _artifact_digest(probe["activations"]),
               "limitation": "Local freeze is an integrity aid; external registration is still required for confirmatory claims."}
    atomic_text(output_path, json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return dict(payload, output=str(output_path))


def _paired_grid_changes(reference, intervention):
    """Keep invalidation and improvement visible on exactly shared trial/item IDs."""
    from .core import read_jsonl
    base = {r["trial"]["trial_id"]: r for r in read_jsonl(Path(reference["dir"]) / "records.jsonl")}
    changed = {r["trial"]["trial_id"]: r for r in read_jsonl(Path(intervention["dir"]) / "records.jsonl")}
    if set(base) != set(changed):
        raise ValueError("Paired steering runs do not share the same task roster")
    transitions, deltas = {}, {}
    for key in sorted(base):
        a, b = base[key]["result"], changed[key]["result"]
        status = a["status"] + "->" + b["status"]
        transitions[status] = transitions.get(status, 0) + 1
        if a["status"] == b["status"] == "ok":
            for metric, value in a["payload"].items():
                if isinstance(value, (int, float)):
                    deltas.setdefault(metric, []).append(float(b["payload"][metric]) - float(value))
    result = {"n_paired_trials": len(base), "validity_transitions": transitions,
              "behavior_mean_change_both_valid": {k: sum(v) / len(v) for k, v in deltas.items()},
              "behavior_n_both_valid": {k: len(v) for k, v in deltas.items()}}
    if reference["capability"] and intervention["capability"]:
        a = _read_json_file(reference["capability"]["path"])
        b = _read_json_file(intervention["capability"]["path"])
        if a["items_sha256"] != b["items_sha256"]:
            raise ValueError("Paired capability materials changed between cells")
        before = {(r["language"], r["item_id"]): r for r in a["records"]}
        after = {(r["language"], r["item_id"]): r for r in b["records"]}
        if before.keys() != after.keys():
            raise ValueError("Paired capability item roster mismatch")
        result["capability"] = {"n_translated_responses": len(before),
            "n_independent_authored_items": len({key[1] for key in before}),
            "correct_to_incorrect": sum(before[k]["correct"] and not after[k]["correct"] for k in before),
            "incorrect_to_correct": sum(not before[k]["correct"] and after[k]["correct"] for k in before),
            "valid_to_invalid": sum(before[k]["valid"] and not after[k]["valid"] for k in before),
            "invalid_to_valid": sum(not before[k]["valid"] and after[k]["valid"] for k in before)}
    return result


def steering_grid(trials_path, output_dir, *, model, revision=None, device="cpu", layers, strengths,
                  positions=("all",), random_seeds=(42,), temperature=0.0, max_output_tokens=512, seed=42,
                  ridge=1.0, limit=None, capability=True, capability_items_path=None, capability_limit=None,
                  evaluation_split="dev", frozen_selection=None):
    """Fit train directions and explore dev behavior; test exactly one frozen choice.

    Learned/random doses match diagonal-standardized train activation displacement.
    Report paired validity, behavior and draft-capability changes. Completion is
    separate from control coverage; outputs do not establish mechanism specificity.
    Matching source/runtime/material/model/generation hashes permit cache reuse.
    """
    from .core import atomic_text, read_jsonl, run_trials, validate_trials, write_jsonl
    from .protocol import runtime_identity
    if evaluation_split not in ("dev", "test"):
        raise ValueError("Steering evaluation_split must be dev or test")
    frozen = _read_json_file(frozen_selection) if frozen_selection else None
    if evaluation_split == "test" and not frozen:
        raise ValueError("Test evaluation requires a frozen dev selection")
    if evaluation_split == "dev" and frozen:
        raise ValueError("A frozen selection is only used for test evaluation")
    layers = _grid_sequence("layers", layers, _int_value("layers"))
    strengths = _grid_sequence("strengths", strengths, _strength_value)
    positions = _grid_sequence("positions", positions, str)
    random_seeds = _grid_sequence("random_seeds", random_seeds, _int_value("random_seeds"))
    if not layers or not strengths or not positions:
        raise ValueError("layers, strengths and positions must be nonempty.")
    if any(position not in _STEERING_POSITIONS for position in positions) or len(set(positions)) != len(positions):
        raise ValueError(f"positions must be distinct values of {_STEERING_POSITIONS}.")
    if len({_strength_name(s) for s in strengths}) != len(strengths):
        raise ValueError("strengths must be distinct.")
    if any(s < 0 for s in random_seeds) or len(set(random_seeds)) != len(random_seeds):
        raise ValueError("random_seeds must be distinct nonnegative integers.")
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("ridge must be finite and strictly positive.")
    all_trials = read_jsonl(trials_path)
    validate_trials(all_trials)
    probe_rows = _probe_trials(all_trials)
    valences = sorted({row["event_valence"] for row in probe_rows})
    if len(valences) != 1:
        raise ValueError("Fit one event_valence at a time: subset whole trial pairs to auspicious or avoidance first.")
    for split in ("train", "dev", "test"):
        conditions = {row["condition"] for row in probe_rows if row["split"] == split}
        if conditions != set(_PROBE_CONDITIONS):
            raise ValueError(f"Split {split!r} must contain both event and neutral trials.")
    test_trials = [row for row in all_trials if row["split"] == evaluation_split]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    test_path = directory / f"{evaluation_split}_trials.jsonl"
    write_jsonl(test_path, test_trials)

    torch, tokenizer, network = runtime = _load_runtime(model, revision, device)
    count = len(_decoder_layers(network))
    resolved = [_layer_index(layer, count) for layer in layers]
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"layers resolve to duplicate blocks: {resolved}.")
    local_digest = _local_model_digest(model)
    shared = {"runtime": runtime, "local_model_sha256": local_digest}
    trials_sha = _trials_digest(probe_rows)
    provenance = {index: _provenance(network, model, revision, index, local_model_sha256=local_digest, tokenizer=tokenizer)
                  for index in resolved}
    activation_path = {index: directory / f"L{index}" / "activations.npz" for index in resolved}
    probe_path = {index: directory / f"L{index}" / "probe.npz" for index in resolved}
    if frozen:
        if (frozen.get("kind") != "frozen_steering_selection" or resolved != [frozen.get("layer")]
                or strengths != [frozen.get("strength")] or positions != [frozen.get("position")]):
            raise ValueError("Test evaluation must use exactly the frozen layer, position and strength")
        expected_settings = {"trials_sha256": _artifact_digest(trials_path), "model": model, "revision": revision,
                             "device": device,
                             "temperature": temperature, "max_output_tokens": max_output_tokens, "seed": seed,
                             "ridge": float(ridge), "random_seeds": random_seeds, "runtime": runtime_identity(),
                             "capability_limit": capability_limit,
                             "capability_items_sha256": _load_capability(capability_items_path)[1]}
        if limit is not None or not capability or not random_seeds:
            raise ValueError("Frozen test evaluation requires full tasks, random controls and capability checks")
        if any(frozen["settings"].get(k) != v for k, v in expected_settings.items()):
            raise ValueError("Frozen selection source/model/runtime/generation/material settings changed")
        if _artifact_digest(frozen["dev_grid_path"]) != frozen["dev_grid_sha256"]:
            raise ValueError("Frozen dev grid hash mismatch")
        for key, destination in (("probe", probe_path[resolved[0]]), ("activation", activation_path[resolved[0]])):
            source = Path(frozen[key + "_path"])
            if _artifact_digest(source) != frozen[key + "_sha256"]:
                raise ValueError("Frozen steering artifact hash mismatch")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copyfile(source, destination)

    stale = []
    for index in resolved:
        expected = dict(provenance[index], kind="prompt_activations", trials_sha256=trials_sha)
        metadata = _artifact_metadata(activation_path[index]) if activation_path[index].exists() else None
        if metadata is None or any(metadata.get(key) != value for key, value in expected.items()):
            stale.append(index)
    if stale:
        if frozen:
            raise ValueError("Frozen activation provenance changed; do not refit after selection")
        captured = _capture_activations(torch, tokenizer, network, device, probe_rows, stale,
                                        provenance[stale[0]]["width"])
        for index in stale:
            _save_activations(activation_path[index], probe_rows, captured[index], provenance[index],
                              len(all_trials) - len(probe_rows))
        del captured

    probes = {}
    for index in resolved:
        activation_sha = _artifact_digest(activation_path[index])
        metadata = _artifact_metadata(probe_path[index]) if probe_path[index].exists() else None
        if (metadata is not None and metadata.get("kind") == "ridge_condition_probe"
                and metadata.get("activation_sha256") == activation_sha
                and metadata.get("report", {}).get("evaluation_split") == ("dev" if frozen else evaluation_split)
                and metadata.get("report", {}).get("ridge") == float(ridge)):
            report = dict(metadata["report"], output=str(probe_path[index]))
            reused = True
        else:
            if frozen:
                raise ValueError("Frozen probe metadata changed; do not refit after selection")
            report = fit_probe(activation_path[index], probe_path[index], ridge=ridge, evaluation_split=evaluation_split)
            reused = False
        probes[index] = {"report": report, "reused": reused}

    settings = {
        "trials_path": str(trials_path), "model": model, "revision": revision, "device": device,
        "layers_requested": layers, "layers_resolved": resolved, "strengths": strengths,
        "positions": positions, "random_seeds": random_seeds, "temperature": temperature,
        "max_output_tokens": max_output_tokens, "seed": seed, "ridge": float(ridge), "limit": limit,
        "capability": capability, "capability_items_path": capability_items_path,
        "capability_limit": capability_limit,
        "trials_sha256": _artifact_digest(trials_path), "runtime": runtime_identity(),
        "capability_items_sha256": _load_capability(capability_items_path)[1] if capability else None,
        "steering_dose": "activation_std",
    }
    runs = []
    manifest = {
        "schema_version": 1, "kind": "steering_grid", "complete": False, "settings": settings,
        "event_valence": valences[0], "test_only": evaluation_split == "test", "evaluation_split": evaluation_split,
        "controls_complete": bool(random_seeds) and capability,
        "frozen_selection_sha256": _artifact_digest(frozen_selection) if frozen else None,
        "restriction": "Train directions; select behavior on dev; evaluate one frozen setting on test. All outputs remain exploratory without external registration.",
        "random_direction_note": "Random and learned runs match unit displacement in diagonal-standardized train activation space; not full covariance or downstream disruption. Zero strength "
                                 "random runs are skipped because they equal the baseline.",
        "interpretation": "Behavior differences from the zero baseline are descriptive; judge them against the "
                          "random-direction runs and capability checks. No causal mediation is claimed.",
        "probes": {str(index): {"path": str(probe_path[index]), "activations": str(activation_path[index]),
                                "reused": probes[index]["reused"],
                                "sha256": _artifact_digest(probe_path[index]),
                                "evaluation_accuracy": probes[index]["report"]["splits"][probes[index]["report"]["evaluation_split"]]["accuracy"],
                                "text_baseline_evaluation_accuracy":
                                    probes[index]["report"]["probe_vs_text_baseline"]["text_baseline_accuracy"],
                                "probe_vs_text_baseline": probes[index]["report"]["probe_vs_text_baseline"]["difference"]}
                   for index in resolved},
        "runs": runs,
    }

    def execute(index, kind, position, strength, random_seed, run_dir):
        provider = OpenWeightProvider(
            model, revision, device, temperature, max_output_tokens, seed,
            steering_path=str(probe_path[index]), strength=strength, layer=index,
            random_direction_seed=random_seed, steering_positions=position, steering_dose="activation_std", _shared=shared)
        summary = run_trials(test_path, run_dir, provider, concurrency=1, limit=limit)
        row = {"layer": index, "kind": kind, "position": position, "strength": strength,
               "random_seed": random_seed, "dir": str(run_dir), "executed": summary["executed"],
               "stored": summary["stored"], "selected": summary["selected"],
               "counts": summary["counts"], "capability": None}
        if capability:
            target = Path(run_dir) / "capability_check.json"
            current = _capability_report(target, provider, items_path=capability_items_path,
                                         limit=capability_limit)
            row["capability"] = {"path": str(target), "overall": current["overall"],
                                 "per_language": current["per_language"]}
        runs.append(row)
        if kind != "baseline":
            baseline = next(r for r in runs if r["layer"] == index and r["kind"] == "baseline")
            row["paired_vs_baseline"] = _paired_grid_changes(baseline, row)
        if kind == "random":
            learned = next(r for r in runs if r["layer"] == index and r["kind"] == "learned"
                           and r["position"] == position and r["strength"] == strength)
            row["paired_vs_learned"] = _paired_grid_changes(learned, row)

    try:
        for index in resolved:
            base = directory / f"L{index}"
            execute(index, "baseline", "all", 0.0, None, base / "baseline")
            for position in positions:
                for strength in strengths:
                    execute(index, "learned", position, strength, None,
                            base / position / f"s{_strength_name(strength)}")
                for random_seed in random_seeds:
                    for strength in strengths:
                        if strength == 0:
                            continue
                        execute(index, "random", position, strength, random_seed,
                                base / position / f"random{random_seed}" / f"s{_strength_name(strength)}")
        manifest["complete"] = True
    finally:
        manifest["n_runs"] = len(runs)
        atomic_text(directory / "grid.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return dict(manifest, output=str(directory / "grid.json"))
