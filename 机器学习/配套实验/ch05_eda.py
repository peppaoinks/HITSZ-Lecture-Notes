"""实验5.2：训练集探索、固定分箱、多图与可追溯报告。"""
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 在导入pyplot之前选择无需窗口的后端。
import matplotlib.pyplot as plt
from ch05_tables import ROOT, load_tables, train_summary


def main():
    raw, clean, audit, duplicates = load_tables()
    train, grouped = train_summary(clean)
    output = ROOT / "ch05_results"
    output.mkdir(exist_ok=True)
    weight = train["weight_kg"].dropna()
    q1, median, q3 = weight.quantile([0.25, 0.5, 0.75]).tolist()
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flagged = train.loc[(train["weight_kg"] < lower) | (train["weight_kg"] > upper), "id"].tolist()
    pairs = train[["weight_kg", "fee_yuan"]].dropna()
    correlation = float(pairs["weight_kg"].corr(pairs["fee_yuan"]))
    counts, edges = np.histogram(weight, bins=[0, 5, 10, 15, 20, 25])
    # 单独记录转换规则；这里不训练模型，也不以测试标签选择规则。
    imputed = clean.copy()
    imputed["weight_missing"] = imputed["weight_kg"].isna().astype(int)
    imputed["weight_kg"] = imputed["weight_kg"].fillna(median)
    summary = {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "matplotlib": matplotlib.__version__},
        "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((ROOT / "ch05_data").glob("*.csv"))},
        "rows_raw": len(raw), "rows_unique": len(clean), "duplicates": duplicates,
        "training_rows": len(train), "valid_weights": len(weight),
        "weight_mean": float(weight.mean()), "weight_median": median,
        "q1": q1, "q3": q3, "fences": [lower, upper], "flagged_ids": flagged,
        "correlation": correlation, "correlation_pairs": len(pairs),
        "histogram_counts": counts.tolist(), "histogram_edges": edges.tolist(),
        "late_counts": {str(int(k)): int(v) for k, v in train["late"].value_counts().items()},
        "policy": "training-only EDA; retain flagged t08; train median fill=4; no model evaluation"
    }
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), layout="constrained")
    color = "#35566E"
    axes[0, 0].scatter(pairs["weight_kg"], pairs["fee_yuan"], color=color)
    axes[0, 0].set(xlabel="Weight (kg)", ylabel="Fee (yuan)", title="Training: complete pairs (n=6)")
    axes[0, 1].hist(weight, bins=edges, color=color, edgecolor="white")
    axes[0, 1].set(xlabel="Weight (kg)", ylabel="Count", title="Training: valid weights (n=6)")
    axes[1, 0].bar(grouped.index, grouped["mean_fee"], color=color)
    axes[1, 0].set(xlabel="Area", ylabel="Mean fee (yuan)", title="Training: 4 rows per area", ylim=(0, 24))
    ordered = train.sort_values("weight_kg").dropna(subset=["weight_kg"])
    axes[1, 1].plot(ordered["weight_kg"], ordered["fee_yuan"], "o-", color=color)
    axes[1, 1].set(xlabel="Weight (kg)", ylabel="Fee (yuan)", title="Sorted observations; no time axis")
    fig.savefig(output / "overview.png", dpi=180)
    plt.close(fig)
    audit.to_csv(output / "audit.csv", index=False)
    grouped.to_csv(output / "groups.csv")
    imputed.to_csv(output / "prepared.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    report = f"""# 包裹教学数据探索报告

问题：重量与费用怎样共同变化？区域均值差能否直接解释为区域作用？

原始{len(raw)}行，完全重复{duplicates}行，唯一记录{len(clean)}行；训练8行、验证2行、测试2行。
训练重量有6个有效值，t04原缺失、t07为非法文本。字段质量核对覆盖各划分；分布分析只用训练集。

重量均值{weight.mean():.6f} kg，中位数{median:g} kg；Q1={q1:g}、Q3={q3:g}，围栏[{lower:g}, {upper:g}]。
t08的20 kg被标记，尚无录入错误证据，保留。有效配对6条，重量与费用相关系数{correlation:.6f}。
数据人为满足线性关系，这不构成真实业务效果或因果证据。训练标签late两类各4条。

区域A/B各4条，平均费用11.5/20.5元，各有3个有效重量。重量构成不同，区域差异不能解释为区域因果效应。
分箱边界0/5/10/15/20/25 kg，频数3/2/0/0/1。最后一箱含右端点。

处理：完全重复记录只留第一条；重量原缺失与转换失败分别写入audit.csv；
填补参数只取训练中位数4 kg，并保留缺失指示列。prepared.csv为教学预处理演示，未训练或评价模型。

限制：样本很小、缺失原因未知、尾部仅1条、数据为构造；不能断言区域效应、缺失机制或预测泛化。
后续需核对采集过程、扩大样本，并按第10章建立无泄漏的预处理流程。

图：overview.png。数字与环境、输入摘要见summary.json。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    assert np.allclose([q1, median, q3, lower, upper], [2.25, 4, 5.75, -3, 11])
    assert flagged == ["t08"] and len(pairs) == 6
    assert counts.tolist() == [3, 2, 0, 0, 1]
    assert np.isclose(correlation, 1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("report and overview saved; all checks passed")


if __name__ == "__main__":
    main()
