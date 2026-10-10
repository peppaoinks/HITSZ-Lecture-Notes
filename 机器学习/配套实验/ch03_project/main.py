"""Run from any working directory: python3 /absolute/path/to/main.py."""
import hashlib
import platform
from pathlib import Path
from io_utils import read_records, write_predictions, write_summary
from metrics import mae
from models import MeanRegressor, NearestRegressor

BASE = Path(__file__).resolve().parent


def columns(records):
    return ([row["weight"] for row in records],
            [row["cost"] for row in records])


def main():
    data_path = BASE / "data" / "parcels.csv"
    records = read_records(data_path)
    groups = {split: [r for r in records if r["split"] == split]
              for split in ("train", "validation", "test")}
    x_train, y_train = columns(groups["train"])
    x_val, y_val = columns(groups["validation"])
    factories = {"mean": MeanRegressor, "nearest": NearestRegressor}
    models = {}
    scores = {}
    for name, factory in factories.items():
        model = factory().fit(x_train, y_train)
        models[name] = model
        scores[name] = mae(y_val, model.predict(x_val))
    # Ties preserve the declared candidate order.
    selected = min(scores, key=scores.get)
    x_test, y_test = columns(groups["test"])
    predicted = models[selected].predict(x_test)
    test_mae = mae(y_test, predicted)
    baseline_mae = mae(y_test, models["mean"].predict(x_test))
    report = {
        "python": platform.python_version(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "split_counts": {name: len(rows) for name, rows in groups.items()},
        "candidate_order": list(factories),
        "selection_rule": "minimum validation MAE; first candidate wins ties",
        "refit_on_train_validation": False,
        "validation_mae": scores, "selected": selected,
        "test_mae": test_mae, "test_baseline_mae": baseline_mae,
    }
    out = BASE / "results"
    out.mkdir(exist_ok=True)
    write_summary(out / "summary.json", report)
    write_predictions(out / "predictions.csv", groups["test"], predicted)
    print("counts:", report["split_counts"])
    for name, score in scores.items():
        print(f"validation {name}: {score:.6f}")
    print("selected:", selected)
    print("test predictions:", predicted)
    print(f"test MAE: {test_mae:.6f}")
    print(f"test baseline MAE: {baseline_mae:.6f}")
    print("saved: results/summary.json, results/predictions.csv")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"experiment failed: {exc}") from exc
