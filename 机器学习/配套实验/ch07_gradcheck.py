"""实验7.2：多路径导数、步长扫描与有限差分检查。"""
import json
import platform
from pathlib import Path
import numpy as np
from ch07_optimization import real_vector
from ch06_svd import loss_and_gradient


def objective(w, alpha=0.05):
    w = real_vector(w)
    if w.shape != (2,) or not np.isfinite(alpha) or alpha < 0:
        raise ValueError("two parameters and nonnegative finite alpha required")
    X = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 2., 2.])
    loss, gradient = loss_and_gradient(X, y, w)
    with np.errstate(over="raise", invalid="raise"):
        loss += float(alpha * np.sum(w**4))
        gradient = gradient + 4 * alpha * w**3
    if not np.isfinite(loss) or not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite objective")
    return loss, gradient


def finite_difference(function, point, h=1e-5):
    point = real_vector(point)
    if not np.isfinite(h) or h <= 0:
        raise ValueError("positive finite h required")
    gradient = np.empty_like(point)
    for j in range(len(point)):
        step = h * max(1.0, abs(point[j]))
        plus, minus = point.copy(), point.copy()
        plus[j] += step
        minus[j] -= step
        if plus[j] == point[j] or minus[j] == point[j]:
            raise ValueError("perturbation below representable resolution")
        f_plus, f_minus = float(function(plus)), float(function(minus))
        if not np.isfinite([f_plus, f_minus]).all():
            raise FloatingPointError("nonfinite function evaluation")
        gradient[j] = (f_plus - f_minus) / (2 * step)
    if not np.isfinite(gradient).all():
        raise FloatingPointError("nonfinite difference")
    return gradient


def direction_checks(w, h=1e-5, alpha=0.05):
    w = real_vector(w)
    _, grad = objective(w, alpha)
    rng = np.random.default_rng(20261013)
    rows = []
    for _ in range(4):
        v = rng.normal(size=w.size)
        v /= np.linalg.norm(v)
        numeric = (objective(w+h*v, alpha)[0] - objective(w-h*v, alpha)[0]) / (2*h)
        analytic = float(grad @ v)
        np.testing.assert_allclose(numeric, analytic, rtol=1e-6, atol=1e-8)
        rows.append({"direction": v.tolist(), "analytic": analytic,
                     "numeric": numeric, "absolute_error": float(abs(analytic-numeric))})
    return rows


def main():
    x = 1.
    a, b = x*x, 3*x
    c = a + b
    full, missing = c*(2*x+3), c*(2*x)
    assert full == 20 and missing == 8
    hs = 10.0 ** -np.arange(1, 17)
    forward = (np.exp(x+hs)-np.exp(x))/hs
    central = (np.exp(x+hs)-np.exp(x-hs))/(2*hs)
    scan = np.column_stack([hs, abs(forward-np.e), abs(central-np.e)])
    w = np.array([0.5, 1.])
    before = w.copy()
    loss, analytic = objective(w)
    numeric = finite_difference(lambda z: objective(z)[0], w)
    np.testing.assert_array_equal(w, before)
    np.testing.assert_allclose(loss, 0.178125, atol=1e-14)
    np.testing.assert_allclose(analytic, [-17/120, 11/30], atol=1e-14)
    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)
    wrong = objective(w, alpha=0)[1]
    assert np.max(np.abs(wrong-numeric)) > 0.19
    summary = {"python": platform.python_version(), "numpy": np.__version__,
               "graph_loss": c*c/2, "graph_derivative": full,
               "missing_branch_derivative": missing,
               "exp_scan_columns": ["h", "forward_error", "central_error"],
               "exp_scan": scan.tolist(), "loss": loss,
               "analytic": analytic.tolist(), "numeric": numeric.tolist(),
               "max_error": float(np.max(np.abs(analytic-numeric))),
               "missing_regularizer_error": (numeric-wrong).tolist(),
               "directions": direction_checks(w)}
    output = Path(__file__).resolve().parent / "ch07_results"
    output.mkdir(exist_ok=True)
    (output / "gradcheck.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(output / "gradcheck.npz", scan=scan, point=w, analytic=analytic, numeric=numeric)
    print(json.dumps(summary, indent=2))
    print("all checks passed")


if __name__ == "__main__":
    main()
