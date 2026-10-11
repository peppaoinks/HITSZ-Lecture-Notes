"""Metrics for nonempty, paired finite numeric sequences."""
from math import isfinite


def mae(actual, predicted):
    if len(actual) != len(predicted) or len(actual) == 0:
        raise ValueError("expected equal, nonempty sequences")
    if not all(isfinite(v) for v in list(actual) + list(predicted)):
        raise ValueError("values must be finite")
    result = sum(abs(p - y) for y, p in zip(actual, predicted)) / len(actual)
    if not isfinite(result):
        raise ValueError("metric overflowed; check data scale")
    return result
