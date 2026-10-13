"""第7章数值答案与有意义的错误检查；不替代正文推导。"""
import numpy as np
from ch07_optimization import quadratic_path
from ch07_gradcheck import objective, finite_difference, direction_checks


def main():
    np.testing.assert_allclose(np.linalg.eigvalsh([[2, 1], [1, 4]]),
                               [3-np.sqrt(2), 3+np.sqrt(2)])
    for eta, expected in [(1/6, [0, 1, 1.5]), (1/2, [0, 3, 1.5])]:
        path, _, _ = quadratic_path([3], [-2], eta, 2)
        np.testing.assert_allclose(path[:, 0]+2, expected)
    for alpha, expected in [(0, [-1/6, 1/6]),
                             (0.05, [-17/120, 11/30]), (0.1, [-7/60, 17/30])]:
        w = np.array([0.5, 1.])
        numeric = finite_difference(lambda z: objective(z, alpha)[0], w)
        np.testing.assert_allclose(objective(w, alpha)[1], expected, atol=1e-14)
        np.testing.assert_allclose(numeric, expected, rtol=1e-6, atol=1e-8)
        direction_checks(w, alpha=alpha)
    for start, expected in [([0., 0.], [1., 1.]), ([2., 0.], [2., 0.])]:
        w = np.array(start)
        for _ in range(80):
            w -= 0.25 * (w.sum()-2)
        np.testing.assert_allclose(w, expected, atol=1e-12)
    h = 0.01
    np.testing.assert_allclose(((1+h)**3-(1-h)**3)/(2*h), 3+h*h)
    assert (abs(h)-abs(-h))/(2*h) == 0  # 对称差分不证明可导
    for bad in [[], [np.nan, 1], [True, False], ["1", "2"], [[1., 2.]]]:
        try:
            finite_difference(lambda z: float(z @ z), bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    for bad_h in [0, -1, np.nan, 1e-30]:
        try:
            finite_difference(lambda z: float(z @ z), [1., 2.], bad_h)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid step accepted")
    print("chapter 7: calculations, rank-deficient limits, directions, and invalid inputs passed")


if __name__ == "__main__":
    main()
