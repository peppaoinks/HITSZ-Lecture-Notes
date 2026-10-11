"""实验5.1：类型、标签对齐、缺失、分组与受约束合并。"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLUMNS = ["id", "split", "area", "weight_kg", "distance_km", "late", "fee_yuan"]
NUMERIC = ["weight_kg", "distance_km", "late", "fee_yuan"]


def load_tables(data_dir=None):
    directory = ROOT / "ch05_data" if data_dir is None else Path(data_dir)
    raw = pd.read_csv(directory / "parcels.csv", dtype="string", keep_default_na=False)
    if list(raw.columns) != COLUMNS or raw.empty:
        raise ValueError("unexpected schema or empty data")
    # 全字段去除首尾空白；保留原始文件，规范化后识别完全重复行。
    raw = raw.apply(lambda column: column.str.strip())
    duplicate_count = int(raw.duplicated().sum())
    clean = raw.drop_duplicates().copy()
    if clean["id"].eq("").any() or clean["id"].duplicated().any():
        raise ValueError("empty id or conflicting records for one id")
    if not clean["split"].isin(["train", "val", "test"]).all():
        raise ValueError("unknown split")
    audit = []
    for name in NUMERIC:
        text = clean[name]
        missing = text.eq("")
        parsed = pd.to_numeric(text.mask(missing), errors="coerce")
        values = parsed.to_numpy(dtype=float, na_value=np.nan)
        invalid = (~missing.to_numpy()) & (~np.isfinite(values))
        for pos in np.flatnonzero(missing.to_numpy() | invalid):
            row = clean.iloc[pos]
            audit.append({"id": row["id"], "column": name,
                          "raw": row[name],
                          "reason": "missing" if missing.iloc[pos] else "invalid"})
        values[~np.isfinite(values)] = np.nan
        clean[name] = values
    if clean[["distance_km", "late", "fee_yuan"]].isna().any().any():
        raise ValueError("only weight may be missing in this teaching schema")
    if (clean[["weight_kg", "distance_km", "fee_yuan"]] < 0).any().any():
        raise ValueError("negative physical quantity")
    if not clean["late"].isin([0, 1]).all():
        raise ValueError("late must be 0 or 1")
    lookup = pd.read_csv(directory / "areas.csv", dtype="string", keep_default_na=False)
    if list(lookup.columns) != ["area", "area_name"]:
        raise ValueError("unexpected area schema")
    if lookup.eq("").any().any():
        raise ValueError("empty lookup field")
    joined = clean.merge(lookup, on="area", how="left",
                         validate="many_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        raise ValueError("unmatched area")
    joined = joined.drop(columns="_merge")
    return raw, joined, pd.DataFrame(audit), duplicate_count


def train_summary(clean):
    train = clean.loc[clean["split"].eq("train")].copy()
    if train.empty:
        raise ValueError("empty training set")
    grouped = train.groupby("area", dropna=False, sort=True).agg(
        rows=("id", "size"), weights=("weight_kg", "count"),
        mean_fee=("fee_yuan", "mean"), late_rate=("late", "mean"))
    return train, grouped


def main():
    raw, clean, audit, duplicates = load_tables()
    train, grouped = train_summary(clean)
    print("rows raw / unique:", len(raw), len(clean))
    print("exact duplicates:", duplicates)
    print("splits:", clean.groupby("split", sort=True).size().to_dict())
    print("audit:\n" + audit.to_string(index=False))
    print("training groups:\n" + grouped.to_string())
    a = pd.Series([10, 20], index=["a", "b"])
    b = pd.Series([1, 2], index=["b", "a"])
    print("aligned sum:", (a + b).to_dict())
    assert (a + b).to_dict() == {"a": 12, "b": 21}
    assert (len(raw), len(clean), duplicates) == (13, 12, 1)
    assert audit["reason"].tolist() == ["missing", "invalid"]
    assert grouped["weights"].tolist() == [3, 3]
    assert grouped["mean_fee"].tolist() == [11.5, 20.5]
    assert train["late"].value_counts().to_dict() == {0.0: 4, 1.0: 4}
    print("all checks passed")


if __name__ == "__main__":
    main()
