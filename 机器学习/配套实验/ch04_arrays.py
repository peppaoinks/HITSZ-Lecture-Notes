"""Experiment 4.1: shape, broadcasting, views, reduction, stable normalization."""
import numpy as np


def main():
    np.set_printoptions(precision=6, suppress=True)
    a = np.array([[1., 2., 3.], [4., 5., 6.]])
    print("shape ndim size:", a.shape, a.ndim, a.size)
    print("column shapes:", a[:, 1].shape, a[:, 1:2].shape)
    print("sum axis=0:", a.sum(axis=0))
    print("sum axis=1:", a.sum(axis=1))
    print("centered:\n", a - a.mean(axis=0, keepdims=True))
    view = a[:, :2]
    view[0, 0] = 99
    picked = a[[0, 1]]
    picked[0, 0] = -1
    print("after view and advanced-index writes:\n", a)
    print("shares:", np.shares_memory(a, view), np.shares_memory(a, picked))
    y = np.array([1., 2., 3.])
    pred = np.array([[1.], [2.], [3.]])
    print("wrong residual shape:", (pred - y).shape)
    print("wrong MSE:", np.mean((pred - y) ** 2))
    print("correct MSE:", np.mean((pred[:, 0] - y) ** 2))
    logits = np.array([[1000., 1001., 1002.], [-1000., -1000., -1000.]])
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / exp_values.sum(axis=1, keepdims=True)
    print("stable row probabilities:\n", probabilities)
    print("row sums:", probabilities.sum(axis=1))
    r1 = np.random.default_rng(2026).integers(0, 100, size=5)
    r2 = np.random.default_rng(2026).integers(0, 100, size=5)
    print("same seed same draws:", np.array_equal(r1, r2))
    assert (pred - y).shape == (3, 3)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2),
                               rtol=0, atol=1e-12)
    print("all checks passed")


if __name__ == "__main__":
    main()
