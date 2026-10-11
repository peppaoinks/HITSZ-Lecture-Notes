"""实验6.2：SVD、条件数、低秩近似与梯度核对。"""
import json
import platform
from pathlib import Path
import numpy as np


def loss_and_gradient(X, y, w):
    # 教学接口只接受有限、实数、二维X与一维y/w。
    arrays = []
    for obj in (X, y, w):
        value = np.asarray(obj)
        if value.dtype.kind not in "iuf":
            raise ValueError("expected real numeric input")
        value = value.astype(np.float64, copy=True)
        if not np.isfinite(value).all():
            raise ValueError("expected finite input")
        arrays.append(value)
    X, y, w = arrays
    if (X.ndim != 2 or y.ndim != 1 or w.ndim != 1 or
            X.shape != (y.size, w.size) or min(X.shape) == 0):
        raise ValueError("incompatible or empty shape")
    with np.errstate(over="raise", invalid="raise"):
        error = X @ w - y
        loss = float((error @ error) / (2 * len(y)))
        grad = X.T @ error / len(y)
    if not np.isfinite(loss) or not np.isfinite(grad).all():
        raise FloatingPointError("nonfinite result")
    return loss, grad


def main():
    A = np.array([[3., 0.], [0., 1.], [0., 0.]])
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    reconstructed = (U * s) @ Vt
    rank1 = (U[:, :1] * s[:1]) @ Vt[:1]
    np.testing.assert_allclose(reconstructed, A, atol=1e-12)
    error = float(np.linalg.norm(A - rank1, "fro"))
    np.testing.assert_allclose(error, 1, atol=1e-12)
    ill = np.diag([1., 1e-8])
    b = np.array([1., 0.])
    delta = np.array([0., 1e-8])
    x = np.linalg.solve(ill, b)
    changed = np.linalg.solve(ill, b + delta)
    amplification = float((np.linalg.norm(changed-x)/np.linalg.norm(x)) /
                          (np.linalg.norm(delta)/np.linalg.norm(b)))
    X = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 2., 2.])
    w = np.array([0.5, 1.])
    loss, grad = loss_and_gradient(X, y, w)
    h = 1e-6
    finite = np.empty_like(w)
    for j in range(w.size):
        step = np.zeros_like(w)
        step[j] = h
        finite[j] = (loss_and_gradient(X, y, w+step)[0] -
                     loss_and_gradient(X, y, w-step)[0]) / (2*h)
    np.testing.assert_allclose(grad, finite, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(grad, [-1/6, 1/6], atol=1e-12)
    summary = {"python": platform.python_version(), "numpy": np.__version__,
               "singular_values": s.tolist(), "rank1_frobenius_error": error,
               "condition_number": float(np.linalg.cond(ill)),
               "condition_gram": float(np.linalg.cond(ill.T @ ill)),
               "solution_before": x.tolist(), "solution_after": changed.tolist(),
               "relative_amplification": amplification, "loss": loss,
               "analytic_gradient": grad.tolist(), "finite_difference": finite.tolist(),
               "gradient_max_error": float(np.max(np.abs(grad-finite)))}
    output = Path(__file__).resolve().parent / "ch06_results"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    np.savez(output / "svd.npz", A=A, U=U, s=s, Vt=Vt, rank1=rank1)
    print(json.dumps(summary, indent=2))
    print("all checks passed")


if __name__ == "__main__":
    main()
