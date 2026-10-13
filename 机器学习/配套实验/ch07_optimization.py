"""实验7.1：二次函数步长、完整轨迹和最小二乘迭代。"""
import json
import platform
from pathlib import Path
import numpy as np
from ch06_svd import loss_and_gradient


def real_vector(value):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf" or a.ndim != 1 or a.size == 0:
        raise ValueError("expected nonempty real vector")
    a = a.astype(np.float64, copy=True)
    if not np.isfinite(a).all():
        raise ValueError("expected finite values")
    return a


def quadratic_path(diagonal, start, eta, steps):
    diagonal, theta = real_vector(diagonal), real_vector(start)
    if diagonal.shape != theta.shape or np.any(diagonal <= 0):
        raise ValueError("positive curvature with matching dimensions required")
    if not np.isfinite(eta) or eta <= 0:
        raise ValueError("positive finite step required")
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("nonnegative integer steps required")
    path = [theta.copy()]
    with np.errstate(over="raise", invalid="raise"):
        for _ in range(steps):
            theta = theta - eta * (diagonal * theta)
            path.append(theta.copy())
        path = np.array(path)
        losses = np.sum(diagonal * path**2, axis=1) / 2
        norms = np.linalg.norm(path * diagonal, axis=1)
    if not np.isfinite(path).all() or not np.isfinite(losses).all():
        raise FloatingPointError("nonfinite trajectory")
    return path, losses, norms


def main():
    output = Path(__file__).resolve().parent / "ch07_results"
    output.mkdir(exist_ok=True)
    arrays, runs = {}, {}
    for label, eta in [("eta_010", 0.1), ("eta_020", 0.2),
                       ("eta_boundary", 2/9), ("eta_023", 0.23)]:
        path, losses, norms = quadratic_path([1, 9], [1, 1], eta, 30)
        exact = (1 - eta * np.array([1., 9.])) ** np.arange(31)[:, None]
        np.testing.assert_allclose(path, exact, rtol=1e-12, atol=1e-14)
        arrays[label + "_path"] = path
        arrays[label + "_loss"] = losses
        arrays[label + "_grad_norm"] = norms
        runs[label] = {"eta": eta, "first_update": path[1].tolist(),
                       "last": path[-1].tolist(), "last_loss": float(losses[-1])}
    X = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 2., 2.])
    w, history = np.zeros(2), []
    for _ in range(400):
        loss, grad = loss_and_gradient(X, y, w)
        history.append([*w, loss, np.linalg.norm(grad)])
        w = w - 0.5 * grad
    loss, grad = loss_and_gradient(X, y, w)
    expected = np.array([7/6, 0.5])
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(loss, 1/36, atol=1e-14)
    direct = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(w, direct, atol=1e-10)
    eigenvalues = np.linalg.eigvalsh(X.T @ X / len(y))
    arrays["least_squares_history"] = np.array(history)
    arrays["least_squares_final"] = w
    summary = {"python": platform.python_version(), "numpy": np.__version__,
               "quadratic_runs": runs, "least_squares": {
                   "iterations": 400, "eta": 0.5, "parameters": w.tolist(),
                   "loss": loss, "gradient_norm": float(np.linalg.norm(grad)),
                   "hessian_eigenvalues": eigenvalues.tolist()}}
    (output / "optimization.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(output / "optimization.npz", **arrays)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("all checks passed")


if __name__ == "__main__":
    main()
