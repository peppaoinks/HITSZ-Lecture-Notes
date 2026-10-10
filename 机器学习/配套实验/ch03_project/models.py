"""Small teaching models; fit returns self and predict returns a new list."""
from math import isfinite


def validate_values(values):
    if len(values) == 0 or not all(isfinite(v) for v in values):
        raise ValueError("expected nonempty finite values")


class MeanRegressor:
    def __init__(self):
        self.mean_ = None

    def fit(self, x, y):
        validate_values(x)
        validate_values(y)
        if len(x) != len(y):
            raise ValueError("lengths differ")
        self.mean_ = sum(y) / len(y)
        return self

    def predict(self, x):
        if self.mean_ is None:
            raise RuntimeError("fit must be called before predict")
        if not all(isfinite(v) for v in x):
            raise ValueError("inputs must be finite")
        return [self.mean_ for _ in x]


class NearestRegressor:
    def __init__(self):
        self.records_ = None

    def fit(self, x, y):
        validate_values(x)
        validate_values(y)
        if len(x) != len(y):
            raise ValueError("lengths differ")
        self.records_ = tuple(zip(x, y))
        return self

    def predict(self, x):
        if self.records_ is None:
            raise RuntimeError("fit must be called before predict")
        if not all(isfinite(v) for v in x):
            raise ValueError("inputs must be finite")
        return [min(self.records_, key=lambda row: abs(row[0] - q))[1]
                for q in x]
