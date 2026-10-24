"""Opt-in scenario tests; resampling a few observed effects is not a null model."""
from contextlib import contextmanager
from contextvars import ContextVar
import math
import random

MIN_SCENARIOS = 12  # eligibility policy, not evidence of adequate power
ASSUMPTION = "independent_sign_symmetric_scenario_effects_under_null"
_METHOD = ContextVar("inference_method", default="descriptive")


@contextmanager
def inference_scope(method="descriptive"):
    if method not in ("descriptive", "scenario_sign_flip"):
        raise ValueError("Unknown inference method")
    token = _METHOD.set(method)
    try:
        yield
    finally:
        _METHOD.reset(token)


def sign_flip_p(values, *, seed=42, permutations=9999):
    """Two-sided mean test under joint invariance to independent sign changes.

    Exact enumeration through 16 scenarios; otherwise random transformations plus
    the observed arrangement. This tests a symmetry model, not treatment assignment.
    """
    values = list(values)
    if not values or any(not math.isfinite(x) for x in values):
        raise ValueError("Sign-flip inputs must be nonempty finite effects")
    observed = abs(math.fsum(values))
    tolerance = 1e-12 * max(1.0, math.fsum(abs(x) for x in values))
    if len(values) <= 16:
        # Gray-code enumeration changes only one sign per step.
        current = math.fsum(values)
        previous = 0
        extreme = int(abs(current) >= observed - tolerance)
        for i in range(1, 1 << len(values)):
            gray = i ^ (i >> 1)
            changed = gray ^ previous
            index = changed.bit_length() - 1
            current += (-2 if gray & changed else 2) * values[index]
            extreme += abs(current) >= observed - tolerance
            previous = gray
        return extreme / (1 << len(values))
    if permutations < 1:
        raise ValueError("permutations must be positive")
    rng = random.Random(seed)
    extreme = sum(abs(math.fsum(x if rng.getrandbits(1) else -x for x in values)) >= observed - tolerance
                  for _ in range(permutations))
    return (extreme + 1) / (permutations + 1)


def estimate(values, *, inferential=True):
    values = list(values)
    if not values:
        return None, None, None, None
    p = None
    if inferential and _METHOD.get() == "scenario_sign_flip" and len(values) >= MIN_SCENARIOS:
        p = sign_flip_p(values)
    # No percentile interval is presented as a population confidence interval.
    return math.fsum(values) / len(values), None, None, p
