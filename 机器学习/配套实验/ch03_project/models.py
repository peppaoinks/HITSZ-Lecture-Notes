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
        mean = sum(y) / len(y)
        if not isfinite(mean):
            raise ValueError("training mean overflowed; check data scale")
        self.mean_ = mean
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
        predicted = []
        for q in x:
            def distance(row):
                value = abs(row[0] - q)
                if not isfinite(value):
                    raise ValueError("distance overflowed; check data scale")
                return value
            predicted.append(min(self.records_, key=distance)[1])
        return predicted
