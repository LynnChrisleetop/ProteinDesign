"""Day 2.5: 用规则生成 6 条种子设计。

⚠️ 编号体系：本脚本所有突变记法用 *with_M* 1-based（M = 第 1 位）。
   数据集 (Sarkisyan) 用 *skip_M* 编号；01_position_pool.py 输出已转换为 with_M。

策略基于 01_position_pool.py 的发现：
  - 致死黑名单避开（含 Y66 发色团，with_M pos 66）
  - 优先赢家高频改动：S65T, S72A, Q80R, V163A, N105K, F46L
  - 文献加成：A206K（单体化）
  - 数据驱动新发现：K158G（max_ratio ≈ 2.48 × WT；这是 01 在 with_M 体系下报告的真实
    super-boost，**之前的 Q157G 是数据集 skip_M / with_M 编号体系混用导致的误读**）
  - 母本对照：avGFP（赢家偏好）vs sfGFP（指南推荐）

6 条梯度（指南 §3 推荐策略 D）：
  Seq_1 safe-baseline      · avGFP + S65T + S72A
  Seq_2 winner-stack       · avGFP + 5 处赢家高频
  Seq_3 sfGFP-control-minus· sfGFP + 最小扰动 S72A（脱 Exclusion）
  Seq_4 boost-engine       · avGFP + F46L + K158G + V163A
  Seq_5 ml-top1-candidate  · avGFP + 6 处 Top-赢家 + K158G
  Seq_6 high-risk-monomer  · sfGFP + K158G + A206K

输出：
  - outputs/seeds.csv      · 6 条种子（Seq_ID, Strategy, Parent, Mutations, Sequence）
  - outputs/seeds.fasta    · 人类可读 FASTA
  - outputs/02_summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DATA_DIR,
    EXCLUSION_CSV,
    OUTPUTS_DIR,
    WT_FASTA_TXT,
    apply_mutations,
    check_sequence,
    parse_wt_fasta,
    write_fasta,
)


SEED_PLAN = [
    {"Seq_ID": 1, "strategy": "safe-baseline",
     "parent": "avGFP", "mutations": "S65T:S72A",
     "rationale": "保守基线；赢家最高频两改 (9/20 each)"},
    {"Seq_ID": 2, "strategy": "winner-stack",
     "parent": "avGFP", "mutations": "S65T:S72A:Q80R:N105K:V163A",
     "rationale": "稳进；avGFP + 赢家 Top-5 高频改动"},
    {"Seq_ID": 3, "strategy": "sfGFP-control-minus",
     "parent": "sfGFP", "mutations": "S72A",
     "rationale": "对照；sfGFP + 最小扰动 S72A（赢家最高频且 tolerant_rate=1.0），既脱 Exclusion 又保 sfGFP 稳定性"},
    {"Seq_ID": 4, "strategy": "boost-engine",
     "parent": "avGFP", "mutations": "F46L:K158G:V163A",
     "rationale": "增益；F46L 折叠 + K158G (data super-boost 2.48× WT, with_M编号) + V163A"},
    {"Seq_ID": 5, "strategy": "ml-top1-candidate",
     "parent": "avGFP", "mutations": "S65T:S72A:N105Y:S147N:I171S:L178V",
     "rationale": "ML最优；ESM+LightGBM从5000组合中筛出ratio=1.27的最强候选；N105Y+S147N+I171S+L178V均为赢家高频tolerant位点"},
    {"Seq_ID": 6, "strategy": "high-risk-monomer-superboost",
     "parent": "sfGFP", "mutations": "K158G:A206K",
     "rationale": "高风险；sfGFP + K158G 数据 super-boost + 单体化 A206K"},
]


def main() -> int:
    if DATA_DIR is None:
        print("[FATAL] no data dir", file=sys.stderr)
        return 1
    print(f"[info] DATA_DIR = {DATA_DIR}")

    print("\n[1] 读 WT FASTA ...")
    wts = parse_wt_fasta(WT_FASTA_TXT)
    parents = {"avGFP": wts["avGFP"], "sfGFP": wts["sfGFP"]}
    for k, v in parents.items():
        print(f"  {k}  len={len(v)}  {v[:30]}...")

    print("\n[2] 读 Exclusion_List.csv ...")
    excl_df = pd.read_csv(EXCLUSION_CSV)
    col = excl_df.columns[0]
    excl_set = set(excl_df[col].astype(str).str.strip().str.upper().tolist())
    print(f"  loaded {len(excl_set)} unique excluded sequences (col='{col}')")

    print("\n[3] 生成 6 条种子 + 合规预检 ...")
    rows = []
    for plan in SEED_PLAN:
        wt = parents[plan["parent"]]
        seq = apply_mutations(wt, plan["mutations"])
        if seq is None:
            print(f"  [FAIL] Seq_{plan['Seq_ID']} 应用突变失败")
            rows.append({**plan, "Sequence": None, "Length": None,
                         "passes_basic": False, "in_exclusion": None})
            continue

        chk = check_sequence(seq)
        in_excl = seq.upper() in excl_set

        flag = (
            "OK"
            if chk["passes_all"] and not in_excl
            else ("EXCLUDED" if in_excl else "BAD")
        )
        print(f"  Seq_{plan['Seq_ID']:<2}  {plan['strategy']:<28}  "
              f"parent={plan['parent']:<6}  "
              f"len={chk['length']:<3}  "
              f"M={'Y' if chk['starts_with_M'] else 'N'}  "
              f"AA={'Y' if chk['only_standard_aa'] else 'N'}  "
              f"in_excl={'Y' if in_excl else 'N'}  -> {flag}")

        rows.append({
            **plan,
            "Sequence": seq,
            "Length": chk["length"],
            "starts_with_M": chk["starts_with_M"],
            "only_standard_aa": chk["only_standard_aa"],
            "passes_basic": chk["passes_all"],
            "in_exclusion": in_excl,
            "verdict": flag,
        })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUTS_DIR / "seeds.csv", index=False)
    print(f"\n  -> outputs/seeds.csv")

    fasta_dict = {
        f"Seq_{int(r['Seq_ID'])}|{r['strategy']}|{r['parent']}|{r['mutations']}": r["Sequence"]
        for r in rows if r["Sequence"]
    }
    write_fasta(fasta_dict, OUTPUTS_DIR / "seeds.fasta")
    print(f"  -> outputs/seeds.fasta")

    n_ok = int((df["verdict"] == "OK").sum())
    n_excluded = int((df["verdict"] == "EXCLUDED").sum())
    n_bad = int((df["verdict"] == "BAD").sum())

    summary = {
        "n_seeds": len(rows),
        "n_OK": n_ok,
        "n_EXCLUDED": n_excluded,
        "n_BAD": n_bad,
        "n_in_exclusion_list": int(df["in_exclusion"].fillna(False).sum()),
        "exclusion_list_size": len(excl_set),
        "seed_summary": [
            {
                "Seq_ID": int(r["Seq_ID"]),
                "strategy": r["strategy"],
                "parent": r["parent"],
                "mutations": r["mutations"],
                "length": r["Length"],
                "verdict": r["verdict"],
            }
            for r in rows
        ],
    }
    (OUTPUTS_DIR / "02_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  -> outputs/02_summary.json")

    print(f"\n[done] OK={n_ok}/6  EXCLUDED={n_excluded}  BAD={n_bad}")
    if n_excluded > 0:
        print("\n  ⚠️  有种子在 Exclusion_List 中，需要扰动或换设计：")
        for r in rows:
            if r["verdict"] == "EXCLUDED":
                print(f"    Seq_{r['Seq_ID']} ({r['strategy']})  父本={r['parent']}  突变={r['mutations']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
