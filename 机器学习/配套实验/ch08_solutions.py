"""第8章手算、概率边界与配对不变性核对。"""
import math
import numpy as np
from ch08_probability import information, bayes_positive, wilson, bernoulli_nll
from ch08_uncertainty import ERROR_A, ERROR_B, paired_bootstrap


def main():
    np.testing.assert_allclose(.54/.82, 27/41)
    np.testing.assert_allclose(bayes_positive(.01, .9, .01), 10/21)
    np.testing.assert_allclose(.138+.0096, .18*.82)
    np.testing.assert_allclose(information([1, 0], [.5, .5]), [0, math.log(2), math.log(2)])
    assert math.isinf(information([1, 0], [0, 1])[2])
    assert bernoulli_nll(1, 5, 5) == 0
    assert bernoulli_nll(0, 0, 5) == 0
    assert math.isinf(bernoulli_nll(1, 4, 5))
    lower, upper = wilson(20, 20)
    np.testing.assert_allclose(lower, .8388748419471806, atol=1e-12)
    np.testing.assert_allclose(upper, 1, atol=1e-14)
    a = paired_bootstrap(ERROR_A, ERROR_B)
    b = paired_bootstrap(ERROR_B, ERROR_A)
    np.testing.assert_array_equal(a, -b)
    ci = np.quantile(a, [.025, .975], method="linear")
    np.testing.assert_allclose(np.quantile(b, [.025, .975]), -ci[::-1], atol=1e-14)
    diff = ERROR_A-ERROR_B
    np.testing.assert_allclose([diff.mean(), diff.var()], [.4, .54])
    index = np.array([0, 0, 2, 3])
    small_a, small_b = np.array([1, 2, 0, 3]), np.array([1, 1, 1, 2])
    assert (small_a[index]-small_b[index]).mean() == 0
    for bad in [[], [-.1, 1.1], [0, 0], [np.nan, 1], [True, False], [[.5, .5]], [".5", ".5"]]:
        try:
            information(bad, [.5, .5])
        except ValueError:
            pass
        else:
            raise AssertionError("invalid probabilities accepted")
    for a_bad, b_bad in [([1, 2], [1]), ([-1, 1], [1, 1]), ([1], [1])]:
        try:
            paired_bootstrap(a_bad, b_bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair accepted")
    try:
        bayes_positive(0, 1, 0)
    except ValueError:
        pass
    else:
        raise AssertionError("zero evidence accepted")
    print("chapter 8: hand calculations, zero probabilities, Wilson, paired symmetry, and invalid inputs passed")


if __name__ == "__main__":
    main()
