"""Strict CSV input and explicit output formats for the teaching project."""
import csv
import json
from math import isfinite

FIELDS = ["id", "split", "weight", "cost"]
SPLITS = ("train", "validation", "test")


def read_records(path):
    records = []
    seen = set()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError(f"expected header {FIELDS}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f"row {line}: wrong field count")
            ident = row["id"].strip()
            split = row["split"].strip()
            if not ident or ident in seen:
                raise ValueError(f"row {line}: empty or duplicate id")
            if split not in SPLITS:
                raise ValueError(f"row {line}: unknown split")
            try:
                weight = float(row["weight"])
                cost = float(row["cost"])
            except ValueError as exc:
                raise ValueError(f"row {line}: invalid number") from exc
            if not all(isfinite(v) and v >= 0 for v in (weight, cost)):
                raise ValueError(f"row {line}: expected finite nonnegative values")
            seen.add(ident)
            records.append({"id": ident, "split": split,
                            "weight": weight, "cost": cost})
    for split in SPLITS:
        if not any(row["split"] == split for row in records):
            raise ValueError(f"empty split: {split}")
    return records


def write_summary(path, report):
    with path.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_predictions(path, records, predicted):
    if len(records) != len(predicted):
        raise ValueError("prediction count differs")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "actual", "predicted", "absolute_error"])
        for row, value in zip(records, predicted):
            writer.writerow([row["id"], row["cost"], value,
                             abs(value - row["cost"])])
