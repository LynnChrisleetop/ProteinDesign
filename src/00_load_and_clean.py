"""Day 1: 数据载入与基线核对。

做四件事：
  1. 解析 5 条 WT FASTA，做长度/起始/字符合规检查。
  2. 把 5 条 WT 落盘为 FASTA（data/processed/wt.fasta）和 CSV。
  3. 读 GFP_data.xlsx 的 brightness 表：列名、各 GFP 类型样本量、WT 行的亮度基线。
  4. 读 beforetopseqs 表：往届高分序列，做合规检查。

输出：
  - data/processed/wt.fasta              · 5 条 WT 单一 FASTA
  - data/processed/wt_summary.csv        · WT 名称 / 长度 / 合规
  - data/processed/before_top_seqs.csv   · 往届 Top 序列 + 长度
  - outputs/00_summary.json              · 关键基线指标，给后续脚本读
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DATA_DIR,
    GFP_DATA_XLSX,
    WT_FASTA_TXT,
    OUTPUTS_DIR,
    PROCESSED_DIR,
    check_sequence,
    fmt_check,
    parse_wt_fasta,
    write_fasta,
)


def step_wt() -> dict:
    print("\n[1/3] 解析 5 条 WT FASTA ...")
    print(f"      源文件: {WT_FASTA_TXT}")
    seqs = parse_wt_fasta(WT_FASTA_TXT)

    rows = []
    print(f"\n  {'name':<10} {'len':>4}   start  chars   verdict")
    print("  " + "-" * 60)
    for name, seq in seqs.items():
        chk = check_sequence(seq)
        verdict = "OK" if chk["passes_all"] else "FAIL"
        print(f"  {name:<10} {chk['length']:>4}   {fmt_check(chk):<28} {verdict}")
        rows.append({
            "name": name,
            "length": chk["length"],
            "starts_with_M": chk["starts_with_M"],
            "length_ok": chk["length_ok"],
            "only_standard_aa": chk["only_standard_aa"],
            "bad_chars": "".join(chk["bad_chars"]) if chk["bad_chars"] else "",
            "passes_all": chk["passes_all"],
            "sequence": seq,
        })

    df = pd.DataFrame(rows)
    df.drop(columns=["sequence"]).to_csv(PROCESSED_DIR / "wt_summary.csv", index=False)
    write_fasta(seqs, PROCESSED_DIR / "wt.fasta")
    print(f"\n  -> wrote {PROCESSED_DIR / 'wt.fasta'}")
    print(f"  -> wrote {PROCESSED_DIR / 'wt_summary.csv'}")
    return {
        "wt_sequences": seqs,
        "wt_summary": rows,
        "n_wt_compliant": int(df["passes_all"].sum()),
    }


def step_brightness() -> dict:
    print("\n[2/3] 读 GFP_data.xlsx :: brightness ...")
    print(f"      源文件: {GFP_DATA_XLSX}")
    df = pd.read_excel(GFP_DATA_XLSX, sheet_name="brightness")
    print(f"      shape={df.shape}  cols={list(df.columns)}")

    if "Brightness" not in df.columns or "GFP type" not in df.columns:
        raise RuntimeError(f"brightness 表列名不符预期: {df.columns.tolist()}")

    type_counts = df["GFP type"].value_counts().to_dict()
    print("\n  按 GFP type 计数：")
    for k, v in type_counts.items():
        print(f"    {k:<12}  {v:>7}")

    wt_rows = df[df["aaMutations"].astype(str).str.upper() == "WT"]
    print(f"\n  WT 行数 = {len(wt_rows)}")
    wt_brightness = {}
    for _, r in wt_rows.iterrows():
        wt_brightness[str(r["GFP type"])] = float(r["Brightness"])
        print(f"    WT({r['GFP type']:<6}) Brightness = {r['Brightness']:.4f}")

    n_mut = df["aaMutations"].astype(str).str.count(":").fillna(0).astype(int) + 1
    n_mut = n_mut.where(df["aaMutations"].astype(str).str.upper() != "WT", 0)
    print("\n  突变层数分布：")
    print(n_mut.value_counts().sort_index().head(10).to_string())

    return {
        "brightness_shape": list(df.shape),
        "brightness_cols": list(df.columns),
        "gfp_type_counts": {str(k): int(v) for k, v in type_counts.items()},
        "wt_brightness_per_type": wt_brightness,
        "brightness_scale_hint": (
            "看起来是 log10 尺度（WT≈3.7 对应原值 ~5000）。"
            "后续训练时若需线性尺度，需做 10**Brightness 反变换。"
        ),
    }


def step_before_top() -> dict:
    print("\n[3/3] 读 GFP_data.xlsx :: beforetopseqs ...")
    df = pd.read_excel(GFP_DATA_XLSX, sheet_name="beforetopseqs")
    print(f"      shape={df.shape}  cols={list(df.columns)}")

    rows = []
    fail = 0
    for i, r in df.iterrows():
        seq = str(r["sequence"]).strip().upper()
        chk = check_sequence(seq)
        rows.append({
            "idx": int(i),
            "year": r.get("year"),
            "length": chk["length"],
            "passes_all": chk["passes_all"],
            "verdict": fmt_check(chk),
        })
        if not chk["passes_all"]:
            fail += 1
    out = pd.DataFrame(rows)
    out.to_csv(PROCESSED_DIR / "before_top_seqs.csv", index=False)
    print(f"  合规 {len(out) - fail}/{len(out)}   -> wrote {PROCESSED_DIR / 'before_top_seqs.csv'}")
    print("\n  长度分布：")
    print(out["length"].describe().to_string())
    return {
        "before_top_shape": list(df.shape),
        "before_top_compliant": int(len(out) - fail),
        "before_top_lengths": out["length"].astype(int).tolist(),
    }


def main() -> int:
    if DATA_DIR is None:
        print("[FATAL] 找不到数据目录。设置 GFP_DATA_DIR 或确保 ../2026Protein Design/ 存在。",
              file=sys.stderr)
        return 1
    print(f"[info] DATA_DIR = {DATA_DIR}")

    summary: dict = {"data_dir": str(DATA_DIR)}
    s1 = step_wt()
    summary.update({
        "wt_compliant_count": s1["n_wt_compliant"],
        "wt_lengths": {n: len(s) for n, s in s1["wt_sequences"].items()},
    })

    s2 = step_brightness()
    summary.update(s2)

    s3 = step_before_top()
    summary.update(s3)

    out_path = OUTPUTS_DIR / "00_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] 关键指标写入 {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
