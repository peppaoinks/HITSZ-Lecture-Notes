"""实验8.1：联合/条件概率、MLE/MAP、信息量和Wilson区间。"""
import json
import math
import platform
from pathlib import Path
from statistics import NormalDist
import numpy as np


def probability_vector(value):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or a.ndim != 1 or not a.size:
        raise ValueError("nonempty real probability vector required")
    a = a.astype(np.float64, copy=True)
    if not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("probabilities must be finite in [0, 1]")
    if not np.isclose(a.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError("probabilities must sum to one")
    return a


def information(p, q):
    p, q = probability_vector(p), probability_vector(q)
    if p.shape != q.shape:
        raise ValueError("matching class dimensions required")
    mask = p > 0
    entropy = float(-np.sum(p[mask] * np.log(p[mask])))
    if np.any(q[mask] == 0):
        return entropy, math.inf, math.inf
    cross = float(-np.sum(p[mask] * np.log(q[mask])))
    return entropy, cross, cross-entropy


def bayes_positive(prevalence, sensitivity, false_positive):
    values = np.array([prevalence, sensitivity, false_positive], dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
        raise ValueError("probabilities required")
    numerator = prevalence * sensitivity
    evidence = numerator + (1-prevalence)*false_positive
    if evidence == 0:
        raise ValueError("conditioning event has zero probability")
    return numerator/evidence


def wilson(successes, n, confidence=0.95):
    if (isinstance(n, bool) or isinstance(successes, bool) or
            not isinstance(n, int) or not isinstance(successes, int) or
            n <= 0 or not 0 <= successes <= n):
        raise ValueError("integer counts with 0 <= successes <= n required")
    if not 0 < confidence < 1:
        raise ValueError("confidence in (0,1) required")
    z = NormalDist().inv_cdf((1+confidence)/2)
    p = successes/n
    denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z/denominator * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return max(0., center-half), min(1., center+half)


def bernoulli_nll(p, k, n):
    if not isinstance(k, int) or not isinstance(n, int) or not 0 <= k <= n or n == 0:
        raise ValueError("valid counts required")
    if not np.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("p in [0,1] required")
    if (p == 0 and k > 0) or (p == 1 and k < n):
        return math.inf
    return -(k*math.log(p) if k else 0) - ((n-k)*math.log1p(-p) if n-k else 0)


def main():
    joint = np.array([[0.54, 0.06], [0.28, 0.12]])
    regions, labels = joint.sum(axis=1), joint.sum(axis=0)
    conditional_delay = joint[:, 1]/regions
    entropy, cross, kl = information([.75, .25], [.5, .5])
    reverse = information([.5, .5], [.75, .25])[2]
    posterior = bayes_positive(.01, .9, .05)
    mle, map_estimate, posterior_mean = 3/5, 4/7, 5/9
    sample = np.array([1., 2., 2., 3.])
    np.testing.assert_allclose(posterior, 2/13)
    np.testing.assert_allclose(conditional_delay, [.1, .3])
    np.testing.assert_allclose(cross-entropy, kl)
    np.testing.assert_allclose(information([.75, .25], [.75, .25])[2], 0, atol=1e-14)
    assert math.isinf(information([1., 0.], [0., 1.])[1])
    summary = {"python": platform.python_version(), "numpy": np.__version__,
               "joint": joint.tolist(), "region_probabilities": regions.tolist(),
               "delay_probability": float(labels[1]),
               "conditional_delay": conditional_delay.tolist(),
               "B_given_delay": float(joint[1, 1]/labels[1]),
               "bayes_positive": posterior,
               "bayes_prevalence_010": bayes_positive(.1, .9, .05),
               "bayes_false_positive_001": bayes_positive(.01, .9, .01),
               "binomial_two_successes": math.comb(4, 2)*(.25**2)*(.75**2),
               "mle": mle, "total_nll": bernoulli_nll(mle, 3, 5),
               "beta22_map": map_estimate, "beta22_posterior_mean": posterior_mean,
               "gaussian_mean_mle": float(sample.mean()),
               "gaussian_variance_mle": float(sample.var()),
               "unbiased_variance": float(sample.var(ddof=1)),
               "mean_standard_error": float(sample.std(ddof=1)/np.sqrt(sample.size)),
               "entropy": entropy, "cross_entropy": cross, "kl": kl, "reverse_kl": reverse,
               "zero_model_probability": "cross_entropy and KL are +infinity",
               "wilson_20_of_20": list(wilson(20, 20))}
    output = Path(__file__).resolve().parent / "ch08_results"
    output.mkdir(exist_ok=True)
    (output / "probability.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("all checks passed")


if __name__ == "__main__":
    main()
