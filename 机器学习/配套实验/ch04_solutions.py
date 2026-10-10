"""Executable checks for Chapter 4 shape, arithmetic and block experiments."""
import numpy as np
from ch04_numeric import checked_mae, fit_scaler, predict_vectorized


def predict_blocked(train, target, query, k, block_size=2):
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be positive")
    if len(query) == 0:
        raise ValueError("empty query")
    blocks = []
    for start in range(0, len(query), block_size):
        blocks.append(predict_vectorized(train, target,
                                         query[start:start + block_size], k))
    return np.concatenate(blocks)


def main():
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    np.testing.assert_array_equal(a.sum(axis=0), [5, 7, 9])
    np.testing.assert_array_equal(a.sum(axis=1), [6, 15])
    np.testing.assert_array_equal(a @ np.array([1, 0, -1]), [-2, -2])
    y = np.array([1., 2., 3.])
    p = y[:, None]
    assert (p - y).shape == (3, 3)
    np.testing.assert_allclose(np.mean((p - y) ** 2), 4 / 3)
    assert np.mean((p[:, 0] - y) ** 2) == 0
    try:
        checked_mae(y, p)
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch not rejected")
    train = np.array([[0., 1.], [2., 1.], [4., 1.]])
    targets = np.array([0., 4., 8.])
    query = np.array([[1., 1.], [3., 1.], [5., 1.]])
    full = predict_vectorized(train, targets, query, 1)
    for size in [1, 2, 10]:
        np.testing.assert_allclose(predict_blocked(train, targets, query, 1, size), full)
    np.testing.assert_array_equal(full, [0., 4., 8.])
    mean, scale = fit_scaler(train)
    assert scale[0, 1] == 1 and mean[0, 1] == 1
    z = (train - mean) / scale
    np.testing.assert_allclose(z.mean(axis=0), [0, 0], atol=1e-12)
    np.testing.assert_allclose(z.std(axis=0), [1, 0], atol=1e-12)
    for k in [0, 4, True]:
        try:
            predict_vectorized(train, targets, query, k)
        except ValueError:
            pass
        else:
            raise AssertionError("bad k not rejected")
    print("all checks passed")


if __name__ == "__main__":
    main()
