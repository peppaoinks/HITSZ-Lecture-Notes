"""实验6.1：投影、最小二乘、秩亏与最小范数解。"""
import numpy as np


def main():
    np.set_printoptions(precision=6, suppress=True)
    X = np.array([[1., 0.], [1., 1.], [1., 2.]])
    y = np.array([1., 2., 2.])
    w, residuals, rank, singular = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ w
    r = y - fitted
    Q, R = np.linalg.qr(X, mode="reduced")
    w_qr = np.linalg.solve(R, Q.T @ y)
    P = Q @ Q.T
    print("w:", w)
    print("prediction:", fitted)
    print("residual y-Xw:", r)
    print("SSE:", float(r @ r))
    print("X.T @ residual:", X.T @ r)
    print("rank:", rank, "singular values:", singular)
    np.testing.assert_allclose(w, [7/6, 1/2], atol=1e-12)
    np.testing.assert_allclose(r, [-1/6, 1/3, -1/6], atol=1e-12)
    np.testing.assert_allclose(w_qr, w, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ y, fitted, atol=1e-12)
    np.testing.assert_allclose(X.T @ r, 0, atol=1e-12)
    Xd = np.column_stack([X, X[:, 1]])
    wd, rd, rank_d, sd = np.linalg.lstsq(Xd, y, rcond=None)
    alternative = wd + np.array([0., 1., -1.])
    print("rank-deficient w:", wd)
    print("rank-deficient rank / residual-array-size:", rank_d, rd.size)
    print("alternative w:", alternative)
    print("coefficient norms:", np.linalg.norm(wd), np.linalg.norm(alternative))
    np.testing.assert_allclose(wd, [7/6, 1/4, 1/4], atol=1e-12)
    np.testing.assert_allclose(Xd @ wd, fitted, atol=1e-12)
    np.testing.assert_allclose(Xd @ alternative, fitted, atol=1e-12)
    assert rank_d == 2 and rd.size == 0
    assert np.linalg.norm(wd) < np.linalg.norm(alternative)
    # 全零向量没有余弦；不通过添加小数掩盖未定义情形。
    a, b = np.array([1., 2.]), np.array([2., 0.])
    print("inner / distance / cosine:", a @ b, np.linalg.norm(a-b),
          (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print("all checks passed")


if __name__ == "__main__":
    main()
