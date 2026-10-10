"""Experiment 4.2: training-only scaling, vectorized kNN and loop equivalence."""
import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np


def check_matrix(x, name):
    if x.ndim != 2 or 0 in x.shape or not np.isfinite(x).all():
        raise ValueError(f"{name}: expected nonempty finite 2D array")


def fit_scaler(x):
    check_matrix(x, "x")
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, ddof=0, keepdims=True)
    scale = np.where(std == 0, 1.0, std)
    return mean, scale


def predict_loop(train, target, query, k):
    predictions = []
    for row in query:
        distances = []
        for i, sample in enumerate(train):
            distance = 0.0
            for a, b in zip(row, sample):
                distance += (a - b) ** 2
            distances.append((distance, i))
        order = sorted(distances, key=lambda item: item[0])
        predictions.append(sum(target[i] for _, i in order[:k]) / k)
    return np.array(predictions)


def predict_vectorized(train, target, query, k):
    check_matrix(train, "train")
    check_matrix(query, "query")
    if query.shape[1] != train.shape[1]:
        raise ValueError("feature counts differ")
    if target.shape != (train.shape[0],) or not np.isfinite(target).all():
        raise ValueError("target shape or values invalid")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= len(train):
        raise ValueError("invalid k")
    delta = query[:, None, :] - train[None, :, :]
    distance2 = np.sum(delta ** 2, axis=2)
    order = np.argsort(distance2, axis=1, kind="stable")[:, :k]
    return target[order].mean(axis=1)


def checked_mae(y, p):
    if y.ndim != 1 or y.size == 0 or p.shape != y.shape:
        raise ValueError("expected paired nonempty 1D arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("nonfinite metric input")
    return float(np.mean(np.abs(p - y)))


def linear_loop(x, w, b):
    result = []
    for row in x:
        value = b
        for feature, weight in zip(row, w):
            value += feature * weight
        result.append(value)
    return np.array(result)


def timed(call):
    samples = []
    for _ in range(3):
        start = perf_counter()
        call()
        samples.append(perf_counter() - start)
    return median(samples)


def main():
    train = np.array([[0, 10, 1], [1, 20, 1], [2, 10, 1], [3, 20, 1]], dtype=float)
    y_train = np.array([4, 6, 8, 10], dtype=float)
    val = np.array([[.2, 11, 1], [1.2, 19, 1], [2.2, 11, 1], [2.8, 19, 1]])
    y_val = np.array([4.4, 6.4, 8.4, 9.6])
    test = np.array([[.4, 12, 1], [1.6, 12, 1], [2.4, 12, 1], [3.5, 22, 1]])
    y_test = np.array([4.8, 7.2, 8.8, 11.])
    original = train.copy()
    mean, scale = fit_scaler(train)
    z_train, z_val, z_test = [(x - mean) / scale for x in (train, val, test)]
    candidates = [1, 3]
    scores = {}
    for k in candidates:
        p = predict_vectorized(z_train, y_train, z_val, k)
        np.testing.assert_allclose(p, predict_loop(z_train, y_train, z_val, k),
                                   rtol=0, atol=1e-12)
        scores[k] = checked_mae(y_val, p)
    selected = min(candidates, key=lambda k: scores[k])
    predicted = predict_vectorized(z_train, y_train, z_test, selected)
    np.testing.assert_allclose(predicted,
                               predict_loop(z_train, y_train, z_test, selected),
                               rtol=0, atol=1e-12)
    np.testing.assert_array_equal(train, original)
    baseline = np.full(y_test.shape, y_train.mean())
    rng = np.random.default_rng(2026)
    x_bench = rng.normal(size=(12000, 12))
    w = rng.normal(size=12)
    loop_output = linear_loop(x_bench, w, 4.0)
    vector_output = x_bench @ w + 4.0
    np.testing.assert_allclose(loop_output, vector_output, rtol=1e-12, atol=1e-12)
    # Warm-up before measuring; timing is an observation, not an assertion.
    linear_loop(x_bench[:10], w, 4.0)
    _ = x_bench @ w + 4.0
    loop_time = timed(lambda: linear_loop(x_bench, w, 4.0))
    vector_time = timed(lambda: x_bench @ w + 4.0)
    report = {
        "python": platform.python_version(), "numpy": np.__version__,
        "seed": 2026, "mean": mean.tolist(), "scale": scale.tolist(),
        "validation_mae": scores, "selected_k": selected,
        "test_predictions": predicted.tolist(),
        "test_mae": checked_mae(y_test, predicted),
        "test_baseline_mae": checked_mae(y_test, baseline),
        "linear_max_abs_difference": float(np.max(np.abs(loop_output - vector_output))),
        "benchmark_shape": list(x_bench.shape), "timing_repeats": 3,
        "median_loop_seconds": loop_time, "median_vector_seconds": vector_time,
    }
    output = Path(__file__).resolve().parent / "ch04_results"
    output.mkdir(exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    np.savez(output / "arrays.npz", train=train, y_train=y_train, validation=val,
             y_validation=y_val, test=test, y_test=y_test, predictions=predicted,
             mean=mean, scale=scale)
    print("mean:", mean.tolist())
    print("scale:", scale.tolist())
    for k, score in scores.items():
        print(f"validation k={k}: {score:.6f}")
    print("selected k:", selected)
    print("test predictions:", predicted.tolist())
    print(f"test MAE: {report['test_mae']:.6f}")
    print(f"test baseline MAE: {report['test_baseline_mae']:.6f}")
    print(f"loop/vector max difference: {report['linear_max_abs_difference']:.3e}")
    print(f"median seconds: loop={loop_time:.6f} vector={vector_time:.6f}")
    print("saved: ch04_results/summary.json, ch04_results/arrays.npz")
    print("all checks passed")


if __name__ == "__main__":
    main()
