"""Day 2: 构造高质量候选突变位点池。

三股证据合成：
  A. 数据驱动 — avGFP 单点突变 (~4.6k 行) per-position 统计
  B. 文献先验 — superfolder/staygold/sfGFP-vs-avGFP diff
  C. 赢家 diff — beforetopseqs 20 条 vs avGFP/sfGFP WT

输出（outputs/）:
  - position_stats_avGFP.csv      · 每个位点的亮度比值统计
  - lethal_blacklist.csv          · 致死位点（绝对避开）
  - safe_positions.csv            · 容忍突变位点（保活）
  - boost_positions.csv           · 见过增益突变的位点
  - super_boost_positions.csv     · 见过 ≥2× 增益的位点
  - winner_diff_avGFP.csv         · 赢家相对 avGFP 的修改位点
  - winner_diff_sfGFP.csv         · 赢家相对 sfGFP 的修改位点
  - literature_priors.csv         · 文献突变包
  - position_pool.csv             · ⭐ 合并打分位点池（给 04_generate_candidates 用）
  - 01_summary.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DATA_DIR,
    GFP_DATA_XLSX,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    WT_FASTA_TXT,
    parse_mutation_str,
    parse_wt_fasta,
)


LETHAL_RATIO = 0.05
RED_LINE_RATIO = 0.30
TOL_RATIO = 0.50
BOOST_RATIO = 1.50
SUPER_RATIO = 2.0


SUPERFOLDER_MUTATIONS = [
    ("S", 30, "R"), ("Y", 39, "N"), ("F", 64, "L"), ("S", 65, "T"),
    ("F", 99, "S"), ("N", 105, "T"), ("Y", 145, "F"), ("M", 153, "T"),
    ("V", 163, "A"), ("I", 171, "V"), ("A", 206, "V"),
]
STAYGOLD_MONOMERIZE = [("A", 206, "K")]
EXTRA_KNOWN = [
    ("F", 46, "L"),
    ("T", 203, "Y"),
]


def linear(log10_b: float) -> float:
    return float(10 ** log10_b)


def per_position_stats(df_av: pd.DataFrame, wt_seq: str, wt_log10: float) -> pd.DataFrame:
    wt_lin = linear(wt_log10)
    df = df_av[df_av["aaMutations"].astype(str).str.upper() != "WT"].copy()
    df["parsed"] = df["aaMutations"].apply(parse_mutation_str)
    df = df[df["parsed"].apply(lambda x: isinstance(x, list))]
    df["n_mut"] = df["parsed"].apply(len)

    single = df[df["n_mut"] == 1].copy()
    single["pos"] = single["parsed"].apply(lambda x: x[0][1])
    single["orig"] = single["parsed"].apply(lambda x: x[0][0])
    single["new"] = single["parsed"].apply(lambda x: x[0][2])
    single["ratio_linear"] = (10 ** single["Brightness"]) / wt_lin

    rows = []
    for pos, g in single.groupby("pos"):
        if not (1 <= pos <= len(wt_seq)):
            continue
        ratios = g["ratio_linear"].values
        best_idx = g["ratio_linear"].idxmax()
        rows.append({
            "pos": int(pos),
            "wt_aa": wt_seq[pos - 1],
            "n_mutants_seen": int(len(g)),
            "n_substitutions": int(g["new"].nunique()),
            "mean_ratio": float(np.mean(ratios)),
            "median_ratio": float(np.median(ratios)),
            "max_ratio": float(np.max(ratios)),
            "min_ratio": float(np.min(ratios)),
            "n_lethal": int(np.sum(ratios < LETHAL_RATIO)),
            "n_red_line": int(np.sum(ratios < RED_LINE_RATIO)),
            "n_tolerant": int(np.sum(ratios >= TOL_RATIO)),
            "n_boost": int(np.sum(ratios >= BOOST_RATIO)),
            "n_super": int(np.sum(ratios >= SUPER_RATIO)),
            "best_substitution": g.loc[best_idx, "new"],
            "best_ratio": float(g["ratio_linear"].max()),
        })

    out = pd.DataFrame(rows).sort_values("pos").reset_index(drop=True)
    out["lethal_rate"] = out["n_lethal"] / out["n_mutants_seen"]
    out["tolerant_rate"] = out["n_tolerant"] / out["n_mutants_seen"]
    out["boost_rate"] = out["n_boost"] / out["n_mutants_seen"]
    return out


def winner_diff(winners: list[str], wts: dict[str, str]) -> pd.DataFrame:
    rows = []
    for wt_name, wt_seq in wts.items():
        for i, win in enumerate(winners):
            if len(win) != len(wt_seq):
                continue
            for pos in range(1, len(wt_seq) + 1):
                if wt_seq[pos - 1] != win[pos - 1]:
                    rows.append({
                        "wt_name": wt_name,
                        "winner_idx": int(i),
                        "pos": pos,
                        "wt_aa": wt_seq[pos - 1],
                        "winner_aa": win[pos - 1],
                    })
    return pd.DataFrame(rows)


def winner_pos_summary(diff_df: pd.DataFrame, wt_name: str) -> pd.DataFrame:
    sub = diff_df[diff_df["wt_name"] == wt_name]
    rows = []
    for pos, g in sub.groupby("pos"):
        common = Counter(g["winner_aa"]).most_common(3)
        rows.append({
            "pos": int(pos),
            "wt_aa": g["wt_aa"].iloc[0],
            "n_winners_changed": int(g["winner_idx"].nunique()),
            "most_common_winner_aa": common[0][0],
            "most_common_count": int(common[0][1]),
            "top3_aa": ",".join([f"{aa}:{c}" for aa, c in common]),
        })
    return pd.DataFrame(rows).sort_values("n_winners_changed", ascending=False).reset_index(drop=True)


def main() -> int:
    if DATA_DIR is None:
        print("[FATAL] no data dir", file=sys.stderr)
        return 1
    print(f"[info] DATA_DIR = {DATA_DIR}")

    print("\n[1] 解析 WT 序列 ...")
    wts = parse_wt_fasta(WT_FASTA_TXT)
    av_wt, sf_wt = wts["avGFP"], wts["sfGFP"]
    print(f"  avGFP len={len(av_wt)}  sfGFP len={len(sf_wt)}")

    print("\n[2] 读 brightness 表 ...")
    df = pd.read_excel(GFP_DATA_XLSX, sheet_name="brightness")
    df_av = df[df["GFP type"] == "avGFP"].copy()
    wt_log10 = float(df_av[df_av["aaMutations"].astype(str).str.upper() == "WT"]
                     ["Brightness"].iloc[0])
    wt_lin = linear(wt_log10)
    print(f"  avGFP rows={len(df_av)}  WT log10={wt_log10:.4f}  (linear={wt_lin:.1f})")

    print("\n[3] Per-position stats from single-point mutants ...")
    stats = per_position_stats(df_av, av_wt, wt_log10)
    stats.to_csv(OUTPUTS_DIR / "position_stats_avGFP.csv", index=False)
    print(f"  covered {len(stats)}/{len(av_wt)} positions"
          f"  -> outputs/position_stats_avGFP.csv")

    print("\n[4] 派生子池 ...")
    lethal_bl = stats[
        (stats["lethal_rate"] > 0.5)
        | ((stats["n_tolerant"] == 0) & (stats["n_mutants_seen"] >= 3))
    ].copy()
    lethal_bl["reason"] = "lethal_rate>0.5 OR no tolerant mutations seen (n>=3)"
    lethal_bl.to_csv(OUTPUTS_DIR / "lethal_blacklist.csv", index=False)
    print(f"  lethal_blacklist     : {len(lethal_bl):3d} pos")

    safe = stats[(stats["n_tolerant"] >= 1) & (stats["lethal_rate"] < 0.3)].copy()
    safe.to_csv(OUTPUTS_DIR / "safe_positions.csv", index=False)
    print(f"  safe_positions       : {len(safe):3d} pos")

    boost = stats[stats["n_boost"] >= 1].sort_values("max_ratio", ascending=False).copy()
    boost.to_csv(OUTPUTS_DIR / "boost_positions.csv", index=False)
    print(f"  boost_positions      : {len(boost):3d} pos  (max_ratio>={BOOST_RATIO}x)")

    sup = stats[stats["n_super"] >= 1].sort_values("max_ratio", ascending=False).copy()
    sup.to_csv(OUTPUTS_DIR / "super_boost_positions.csv", index=False)
    print(f"  super_boost_positions: {len(sup):3d} pos  (max_ratio>={SUPER_RATIO}x)")

    print("\n[5] 赢家 diff ...")
    win_df = pd.read_excel(GFP_DATA_XLSX, sheet_name="beforetopseqs")
    winners = win_df["sequence"].astype(str).str.strip().str.upper().tolist()
    print(f"  loaded {len(winners)} winners; lengths={sorted(set(len(s) for s in winners))}")

    diff_df = winner_diff(winners, {"avGFP": av_wt, "sfGFP": sf_wt})
    diff_df.to_csv(PROCESSED_DIR / "winner_diff_raw.csv", index=False)

    wsum_av = winner_pos_summary(diff_df, "avGFP")
    wsum_av.to_csv(OUTPUTS_DIR / "winner_diff_avGFP.csv", index=False)
    wsum_sf = winner_pos_summary(diff_df, "sfGFP")
    wsum_sf.to_csv(OUTPUTS_DIR / "winner_diff_sfGFP.csv", index=False)
    print(f"  winner_diff_avGFP    : {len(wsum_av):3d} pos  (top 5 changed)")
    print(wsum_av.head(5).to_string(index=False))
    print(f"\n  winner_diff_sfGFP    : {len(wsum_sf):3d} pos  (top 5 changed)")
    print(wsum_sf.head(5).to_string(index=False))

    sf_vs_av = [
        (pos, av_wt[pos - 1], sf_wt[pos - 1])
        for pos in range(1, min(len(av_wt), len(sf_wt)) + 1)
        if av_wt[pos - 1] != sf_wt[pos - 1]
    ]
    print(f"\n  sfGFP vs avGFP diff   : {len(sf_vs_av)} pos")
    print("    " + ", ".join(f"{a}{p}{b}" for p, a, b in sf_vs_av))

    print("\n[6] 文献先验 ...")
    lit_rows = []
    for tag, lst in [("superfolder", SUPERFOLDER_MUTATIONS),
                     ("staygold", STAYGOLD_MONOMERIZE),
                     ("extra", EXTRA_KNOWN)]:
        for orig, pos, new in lst:
            av_aa = av_wt[pos - 1] if 1 <= pos <= len(av_wt) else "?"
            lit_rows.append({
                "source": tag,
                "mutation": f"{orig}{pos}{new}",
                "pos": pos, "orig_aa": orig, "new_aa": new,
                "avGFP_wt_aa": av_aa, "matches_avGFP": av_aa == orig,
            })
    lit = pd.DataFrame(lit_rows)
    lit.to_csv(OUTPUTS_DIR / "literature_priors.csv", index=False)
    print(f"  literature_priors    : {len(lit)} entries")

    print("\n[7] 合并位点池 ...")
    L = len(av_wt)
    pool = pd.DataFrame({"pos": range(1, L + 1)})
    pool["wt_aa_avGFP"] = [av_wt[i - 1] for i in pool["pos"]]
    pool["wt_aa_sfGFP"] = [sf_wt[i - 1] if i - 1 < len(sf_wt) else "-" for i in pool["pos"]]

    pool = pool.merge(
        stats[["pos", "n_mutants_seen", "lethal_rate", "tolerant_rate",
               "boost_rate", "n_boost", "n_super", "max_ratio", "best_substitution"]],
        on="pos", how="left",
    )

    lethal_set = set(int(p) for p in lethal_bl["pos"].tolist())
    pool["is_lethal"] = pool["pos"].isin(lethal_set)

    pool = pool.merge(
        wsum_av[["pos", "n_winners_changed", "most_common_winner_aa"]],
        on="pos", how="left",
    )
    pool["n_winners_changed"] = pool["n_winners_changed"].fillna(0).astype(int)

    lit_pos = set(int(p) for p in lit["pos"].tolist())
    pool["in_literature"] = pool["pos"].isin(lit_pos)

    sf_diff_pos = set(p for p, _, _ in sf_vs_av)
    pool["is_sfGFP_diff"] = pool["pos"].isin(sf_diff_pos)

    def categorize(row):
        if row["is_lethal"]:
            return "LETHAL"
        cat = []
        if row["n_winners_changed"] >= 3 or (row["n_boost"] or 0) >= 1:
            cat.append("BOOST")
        if row["in_literature"] or row["is_sfGFP_diff"]:
            cat.append("LIT")
        if (row["tolerant_rate"] or 0) >= 0.5 or row["n_winners_changed"] >= 1:
            cat.append("SAFE")
        return "+".join(cat) if cat else "UNKNOWN"
    pool["category"] = pool.apply(categorize, axis=1)

    def score(row):
        if row["is_lethal"]:
            return -100.0
        s = 0.0
        s += min(row["max_ratio"] or 0, 5.0) * 1.0
        s += (row["n_boost"] or 0) * 0.5
        s += row["n_winners_changed"] * 1.5
        s += 2.0 if row["in_literature"] else 0.0
        s += 1.5 if row["is_sfGFP_diff"] else 0.0
        return float(s)
    pool["score"] = pool.apply(score, axis=1)

    pool.to_csv(OUTPUTS_DIR / "position_pool.csv", index=False)
    print(f"  -> outputs/position_pool.csv  ({len(pool)} rows, {L} positions)")

    print("\n  ====== TOP 30 候选位点（按综合得分）======")
    top30 = pool[pool["score"] > 0].sort_values("score", ascending=False).head(30)
    print(top30[["pos", "wt_aa_avGFP", "wt_aa_sfGFP", "category", "score",
                 "n_winners_changed", "most_common_winner_aa",
                 "n_boost", "best_substitution", "max_ratio",
                 "tolerant_rate", "in_literature", "is_sfGFP_diff"]]
          .to_string(index=False))

    cat_counts = pool["category"].value_counts().to_dict()
    print("\n  类别分布：")
    for k, v in cat_counts.items():
        print(f"    {k:<24} {v}")

    summary = {
        "data_dir": str(DATA_DIR),
        "wt_brightness_log10_avGFP": wt_log10,
        "wt_brightness_linear_avGFP": wt_lin,
        "thresholds": {
            "lethal_ratio": LETHAL_RATIO, "red_line_ratio": RED_LINE_RATIO,
            "tolerant_ratio": TOL_RATIO, "boost_ratio": BOOST_RATIO,
            "super_ratio": SUPER_RATIO,
        },
        "n_positions_with_data": int(stats["pos"].nunique()),
        "n_lethal_blacklist": int(len(lethal_bl)),
        "n_safe_pool": int(len(safe)),
        "n_boost_pool": int(len(boost)),
        "n_super_pool": int(len(sup)),
        "n_lit_priors": int(len(lit)),
        "n_winner_diff_avGFP": int(len(wsum_av)),
        "n_winner_diff_sfGFP": int(len(wsum_sf)),
        "sfGFP_vs_avGFP_diff_positions": [
            {"pos": p, "avGFP": a, "sfGFP": b} for p, a, b in sf_vs_av
        ],
        "pool_category_counts": cat_counts,
    }
    (OUTPUTS_DIR / "01_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] -> outputs/01_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
