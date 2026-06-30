"""Day 2.5 收尾：6 条种子的多样性与风险梯度评估。

为什么需要：
  最终排名仅看 Top-1。但 6 条若全在同一母本+同一突变集附近，**集体失败**
  的风险高。这一步检查 6 条之间的差异性，并给出第 4 母本（cgreGFP）
  备选方案（避免 6 槽都依赖 av/sf 同源）。

输出：
  - outputs/03_pairwise_hamming.csv   · 6 条两两 Hamming 距离矩阵
  - outputs/03_diversity_report.json  · 多样性度量与诊断
  - outputs/03_cgreGFP_candidate.txt  · cgreGFP-based 备选种子的草稿（如适用）
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DATA_DIR,
    OUTPUTS_DIR,
    WT_FASTA_TXT,
    check_sequence,
    parse_wt_fasta,
)


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def main() -> int:
    if DATA_DIR is None:
        print("[FATAL] no data dir", file=sys.stderr)
        return 1
    print(f"[info] DATA_DIR = {DATA_DIR}")

    seeds = pd.read_csv(OUTPUTS_DIR / "seeds.csv").dropna(subset=["Sequence"])
    seeds = seeds.reset_index(drop=True)
    print(f"\n[1] 读 {len(seeds)} 条种子 ...")
    print(seeds[["Seq_ID", "strategy", "parent", "mutations"]].to_string(index=False))

    wts = parse_wt_fasta(WT_FASTA_TXT)

    print("\n[2] 两两 Hamming 距离 ...")
    n = len(seeds)
    mat = np.zeros((n, n), dtype=int)
    pairs = []
    for i, j in combinations(range(n), 2):
        d = hamming(seeds.iloc[i]["Sequence"], seeds.iloc[j]["Sequence"])
        mat[i, j] = mat[j, i] = d
        pairs.append({
            "Seq_A": int(seeds.iloc[i]["Seq_ID"]),
            "Seq_B": int(seeds.iloc[j]["Seq_ID"]),
            "strategy_A": seeds.iloc[i]["strategy"],
            "strategy_B": seeds.iloc[j]["strategy"],
            "hamming": d,
        })

    pair_df = pd.DataFrame(pairs).sort_values("hamming")
    pair_df.to_csv(OUTPUTS_DIR / "03_pairwise_hamming.csv", index=False)
    print(pair_df.to_string(index=False))

    print("\n  Hamming 矩阵 (Seq_1..Seq_6):")
    header = "       " + " ".join(f"Seq_{int(s):<2}" for s in seeds["Seq_ID"])
    print(header)
    for i in range(n):
        row = " ".join(f"{mat[i,j]:>5}" for j in range(n))
        print(f"  Seq_{int(seeds.iloc[i]['Seq_ID']):<2}  {row}")

    print("\n[3] 与各 WT 的距离 ...")
    rows_wt = []
    for _, r in seeds.iterrows():
        rec = {"Seq_ID": int(r["Seq_ID"]), "strategy": r["strategy"], "parent": r["parent"]}
        for wt_name, wt_seq in wts.items():
            rec[f"d_to_{wt_name}"] = hamming(r["Sequence"], wt_seq)
        rows_wt.append(rec)
    wt_df = pd.DataFrame(rows_wt)
    print(wt_df.to_string(index=False))
    wt_df.to_csv(OUTPUTS_DIR / "03_dist_to_wt.csv", index=False)

    print("\n[4] 多样性诊断 ...")
    diag = {}
    diag["pairwise_hamming_mean"] = float(pair_df["hamming"].mean())
    diag["pairwise_hamming_median"] = float(pair_df["hamming"].median())
    diag["pairwise_hamming_min"] = int(pair_df["hamming"].min())
    diag["pairwise_hamming_max"] = int(pair_df["hamming"].max())
    diag["n_unique_parents"] = int(seeds["parent"].nunique())
    diag["parents_used"] = seeds["parent"].value_counts().to_dict()
    diag["closest_to_avGFP"] = int(wt_df["d_to_avGFP"].min())
    diag["farthest_to_avGFP"] = int(wt_df["d_to_avGFP"].max())

    if diag["pairwise_hamming_mean"] < 5:
        diag["warning_too_clustered"] = (
            "6 条种子两两 Hamming 平均 < 5，过于聚集，"
            "若赛方实测 avGFP/sfGFP 母本不利则可能集体失败。"
        )
    if diag["pairwise_hamming_min"] <= 2:
        too_close = pair_df[pair_df["hamming"] <= 2].to_dict("records")
        diag["warning_pair_near_duplicate"] = {
            "msg": "有种子对 Hamming ≤ 2，几乎重复，浪费 6 槽",
            "pairs": too_close,
        }
    if diag["n_unique_parents"] < 3:
        diag["warning_few_parents"] = (
            "只使用了 ≤2 个母本（av/sf）；建议至少加 1 条 cgreGFP 或共识母本。"
        )

    print("  pairwise mean/median/min/max =",
          diag["pairwise_hamming_mean"], "/",
          diag["pairwise_hamming_median"], "/",
          diag["pairwise_hamming_min"], "/",
          diag["pairwise_hamming_max"])
    print("  unique parents =", diag["n_unique_parents"], diag["parents_used"])
    for k, v in diag.items():
        if k.startswith("warning"):
            print(f"  ⚠️  {k}: {v}")

    print("\n[5] cgreGFP 备选母本评估 ...")
    cg = wts.get("cgreGFP")
    if cg:
        chk = check_sequence(cg)
        print(f"  cgreGFP WT 长度 {chk['length']}, 合规={chk['passes_all']}")
        print(f"  cgreGFP 与 avGFP Hamming = {hamming(cg, wts['avGFP']):>3} (近似上限{min(len(cg), len(wts['avGFP']))})")
        print(f"  cgreGFP 与 sfGFP Hamming = {hamming(cg, wts['sfGFP']):>3}")
        print(f"  → 信号：cgreGFP 与 av/sf 距离 >100 时，是真正独立的母本")
        diag["cgreGFP_len"] = chk["length"]
        diag["cgreGFP_compliant"] = chk["passes_all"]
        diag["cgreGFP_d_to_avGFP"] = hamming(cg, wts["avGFP"])

        sketch_path = OUTPUTS_DIR / "03_cgreGFP_candidate.txt"
        sketch = (
            "# cgreGFP-based 备选 Seq_6 草稿\n"
            "# 思路: 用 cgreGFP 作母本（数据中 WT 亮度最高 log10=4.50，对应线性 ~31403）\n"
            "# 仅微调 1-2 处保活突变以脱离 Exclusion_List\n"
            "# 注意: cgreGFP 不在 51715 行 avGFP 训练数据里，无 ML 模型支撑；\n"
            "#       但其 WT 亮度数据告诉我们这是个潜在的高亮度母本\n"
            "#\n"
            f"WT_seq cgreGFP (len={chk['length']}):\n"
            f"{cg}\n"
            "\n"
            "# 建议突变 (待用 03 之后的工具确认)：\n"
            "# - 无显著替代基线，暂保留原 sfGFP+Q157G+A206K (Seq_6) 不动\n"
            "# - 若后续 ML 显示 av/sf 模型不可靠，可启用此备选并补 1-2 处保活突变\n"
        )
        sketch_path.write_text(sketch, encoding="utf-8")
        print(f"  -> {sketch_path}")

    diag["diversity_verdict"] = (
        "GOOD" if (
            diag["pairwise_hamming_mean"] >= 5
            and diag["pairwise_hamming_min"] >= 3
            and diag["n_unique_parents"] >= 2
        ) else "NEEDS_DIVERSIFICATION"
    )

    (OUTPUTS_DIR / "03_diversity_report.json").write_text(
        json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] -> outputs/03_diversity_report.json   verdict={diag['diversity_verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
