"""第六章手算与梯度接口的必要边界核对。"""
import numpy as np
from ch06_svd import loss_and_gradient


def main():
    a, b = np.array([1., 2.]), np.array([2., 0.])
    np.testing.assert_allclose([a @ b, np.linalg.norm(a-b)], [2, np.sqrt(5)])
    q = np.array([1., 1., 0.]) / np.sqrt(2)
    P = np.outer(q, q)
    y = np.array([2., 0., 3.])
    np.testing.assert_allclose(P @ y, [1, 1, 0], atol=1e-12)
    np.testing.assert_allclose(q @ (y-P@y), 0, atol=1e-12)
    H = np.array([[2., 1.], [1., 2.]])
    np.testing.assert_allclose(np.linalg.eigvalsh(H), [1, 3], atol=1e-12)
    X = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 2., 2.])
    loss, g = loss_and_gradient(X, y, [0.5, 1.])
    np.testing.assert_allclose(loss, .125, atol=1e-12)
    np.testing.assert_allclose(g, [-1/6, 1/6], atol=1e-12)
    for args in [(X, y[:, None], [0., 0.]), (X, y, [True, False]),
                 (X, y, [np.nan, 0.]), ([], [], []), (X, y, ["0", "0"])]:
        try:
            loss_and_gradient(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    try:
        loss_and_gradient([[1e308]], [0.], [1e308])
    except FloatingPointError:
        pass
    else:
        raise AssertionError("overflow accepted")
    print("projection: [1, 1, 0]; eigenvalues: [1, 3]")
    print("loss: 0.125; gradient: [-1/6, 1/6]")
    print("all checks passed")


if __name__ == "__main__":
    main()
