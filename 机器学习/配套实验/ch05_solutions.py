"""第五章习题数值与失败路径核对。"""
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import numpy as np
import pandas as pd
from ch05_tables import ROOT, load_tables, train_summary


def main():
    _, clean, audit, _ = load_tables()
    train, grouped = train_summary(clean)
    values = pd.Series([1., 2., 3., 5., 6., 20.])
    np.testing.assert_allclose(values.quantile([.25, .5, .75]), [2.25, 4, 5.75])
    assert np.isclose(values.std(ddof=1)**2, 1481/30)
    assert grouped["rows"].tolist() == [4, 4]
    assert grouped["weights"].tolist() == [3, 3]
    left = pd.DataFrame({"key": ["A", "A"], "x": [1, 2]})
    right = pd.DataFrame({"key": ["A", "A"], "z": [10, 20]})
    assert len(left.merge(right, on="key")) == 4
    try:
        left.merge(right, on="key", validate="many_to_one")
    except pd.errors.MergeError:
        pass
    else:
        raise AssertionError("duplicate lookup key accepted")
    # 测试标签改变不能改变训练探索数字。
    changed = clean.copy()
    changed.loc[changed["split"].eq("test"), "fee_yuan"] = 999.
    pd.testing.assert_frame_equal(train_summary(changed)[1], grouped)
    with TemporaryDirectory() as tmp:
        directory = Path(tmp)
        for file in (ROOT / "ch05_data").glob("*.csv"):
            shutil.copyfile(file, directory / file.name)
        raw = pd.read_csv(directory / "parcels.csv", dtype="string", keep_default_na=False)
        raw.loc[8, "fee_yuan"] = "11"  # t03出现冲突，不能按id静默保留第一条。
        raw.to_csv(directory / "parcels.csv", index=False)
        try:
            load_tables(directory)
        except ValueError as error:
            assert "conflicting" in str(error)
        else:
            raise AssertionError("conflicting id accepted")
    print("fences: -3, 11; flag t08; training median: 4")
    print("all checks passed")


if __name__ == "__main__":
    main()
