"""Experiment 3.1: references, copies, mutable defaults, and file round-trip."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def collect(value, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(value)
    return bucket


def main():
    rows = [[1, 6], [2, 8]]
    alias = rows
    shallow = rows.copy()
    deep = copy.deepcopy(rows)
    alias.append([3, 10])
    shallow[0][1] = 99
    print("rows:", rows)
    print("shallow:", shallow)
    print("deep:", deep)
    print("identity:", alias is rows, shallow is rows,
          shallow[0] is rows[0], deep[0] is rows[0])
    print("fresh:", collect(1), collect(2))
    print("captured:", [fn(10) for fn in [lambda x, k=k: x + k
                                        for k in range(3)]])
    with TemporaryDirectory() as folder:
        path = Path(folder) / "report.json"
        with path.open("w", encoding="utf-8") as stream:
            json.dump({"count": 3, "note": "教学数据"}, stream,
                      ensure_ascii=False, allow_nan=False)
        with path.open("r", encoding="utf-8") as stream:
            report = json.load(stream)
        print("round trip:", report["count"], report["note"])
        try:
            float("missing")
        except ValueError as exc:
            print("caught:", type(exc).__name__)
    assert deep == [[1, 6], [2, 8]]
    assert collect(1) == [1] and collect(2) == [2]
    print("all checks passed")


if __name__ == "__main__":
    main()
