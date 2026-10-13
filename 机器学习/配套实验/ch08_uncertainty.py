"""实验8.2：已知方差均值区间覆盖与固定预测的配对bootstrap。"""
import json
import platform
from pathlib import Path
from statistics import NormalDist
import numpy as np
from ch07_optimization import real_vector

SEED = 20261013
ERROR_A = np.array([1, 2, 0, 3, 1, 2, 1, 4, 0, 2, 3, 1, 2, 0, 4, 1, 2, 3, 1, 2], dtype=float)
ERROR_B = np.array([1, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1, 1, 3, 1, 1, 2, 1, 1], dtype=float)


def paired_bootstrap(error_a, error_b, repeats=10000, seed=SEED):
    a, b = real_vector(error_a), real_vector(error_b)
    if a.shape != b.shape or a.size < 2 or np.any(a < 0) or np.any(b < 0):
        raise ValueError("matching nonnegative error vectors with n >= 2 required")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 2:
        raise ValueError("integer repeats >= 2 required")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(a), size=(repeats, len(a)))
    boot = (a-b)[index].mean(axis=1)
    return boot


def main():
    rng = np.random.default_rng(SEED)
    n, repeats = 25, 5000
    samples = rng.normal(10., 2., size=(repeats, n))
    means = samples.mean(axis=1)
    z = NormalDist().inv_cdf(.975)
    half = z*2/np.sqrt(n)
    covered = (means-half <= 10) & (10 <= means+half)
    coverage = float(covered.mean())
    assert .93 < coverage < .97
    assert .37 < means.std(ddof=1) < .43
    boot = paired_bootstrap(ERROR_A, ERROR_B)
    interval = np.quantile(boot, [.025, .975], method="linear")
    difference = ERROR_A-ERROR_B
    expected_se = float(np.sqrt(difference.var()/len(difference)))
    np.testing.assert_allclose(difference.mean(), .4, atol=1e-14)
    assert abs(boot.std(ddof=1)-expected_se) < .01
    summary = {"python": platform.python_version(), "numpy": np.__version__, "seed": SEED,
               "coverage": {"repetitions": repeats, "n": n, "true_mu": 10, "known_sigma": 2,
                            "half_width": float(half), "fraction": coverage,
                            "empirical_mean_sd": float(means.std(ddof=1))},
               "paired_bootstrap": {"repetitions": len(boot), "n": len(difference),
                                    "mae_a": float(ERROR_A.mean()), "mae_b": float(ERROR_B.mean()),
                                    "difference_a_minus_b": float(difference.mean()),
                                    "empirical_difference_variance": float(difference.var()),
                                    "conditional_theoretical_se": expected_se,
                                    "bootstrap_se": float(boot.std(ddof=1)),
                                    "percentile_interval": interval.tolist(), "quantile_method": "linear",
                                    "scope": "fixed predictions; iid test records; no retraining"}}
    output = Path(__file__).resolve().parent / "ch08_results"
    output.mkdir(exist_ok=True)
    (output / "uncertainty.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(output / "uncertainty.npz", means=means, covered=covered,
             error_a=ERROR_A, error_b=ERROR_B, bootstrap_difference=boot)
    print(json.dumps(summary, indent=2))
    print("all checks passed")


if __name__ == "__main__":
    main()
