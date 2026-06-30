"""Day 4 延伸：对 beforetopseqs 20 条往届高分序列做 ThermoMPNN 热稳定评分。

输入：
  - GFP_data.xlsx::beforetopseqs（20 条全长序列）
  - outputs/ThermoMPNN_inference_2WUR.csv（avGFP，默认母本）
  - 可选 outputs/ThermoMPNN_inference_2B3P.csv（sfGFP）

输出：
  - outputs/08d_thermompnn_winner_mutations.csv  每位点 ddG
  - outputs/08d_thermompnn_winners.csv           每条 winner 汇总

说明：
  - 20 条更接近 avGFP，默认相对 avGFP WT 求 diff 并查 2WUR 模型。
  - 多突变 ddG 仍用单点求和近似（与 08_thermompnn.py 一致）。
  - 若已有 ThermoMPNN_inference_*.csv，不会重跑推理。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import GFP_DATA_XLSX, OUTPUTS_DIR, WT_FASTA_TXT, parse_wt_fasta  # noqa: E402

# 复用 08 的 ThermoMPNN 查表与位点映射
from importlib import import_module

_thermo = import_module("08_thermompnn")
ensure_inference_csv = _thermo.ensure_inference_csv
load_ddg_lookup = _thermo.load_ddg_lookup
build_with_m_to_model_pos_map = _thermo.build_with_m_to_model_pos_map


def seq_diff_mutations(wt_seq: str, mut_seq: str) -> list[tuple[str, int, str]]:
    """全长 diff → [(orig, pos_with_M, new), ...]。"""
    wt_seq = wt_seq.strip().upper()
    mut_seq = mut_seq.strip().upper()
    if len(wt_seq) != len(mut_seq):
        raise ValueError(f"length mismatch: wt={len(wt_seq)} mut={len(mut_seq)}")
    out: list[tuple[str, int, str]] = []
    for i, (a, b) in enumerate(zip(wt_seq, mut_seq)):
        if a != b:
            out.append((a, i + 1, b))
    return out


def mutations_to_str(parsed: list[tuple[str, int, str]]) -> str:
    return ":".join(f"{o}{p}{n}" for o, p, n in parsed)


def score_winner(
    winner_idx: int,
    year: object,
    parent: str,
    wt_seq: str,
    win_seq: str,
    lookup: dict,
    with_m_map: dict[int, int],
) -> tuple[list[dict], dict]:
    parsed = seq_diff_mutations(wt_seq, win_seq)
    mut_str = mutations_to_str(parsed)

    mut_rows: list[dict] = []
    ddg_values: list[float] = []
    n_miss = 0

    for orig, pos_with_m, new in parsed:
        base = {
            "winner_idx": winner_idx,
            "year": year,
            "parent": parent,
            "mutation": f"{orig}{pos_with_m}{new}",
            "position_with_M": pos_with_m,
        }
        if pos_with_m not in with_m_map:
            n_miss += 1
            mut_rows.append({
                **base,
                "position_0based": None,
                "ddG_pred": None,
                "status": "position_not_covered_in_pdb",
            })
            continue

        pos0 = with_m_map[pos_with_m]
        key = (pos0, new)
        if key not in lookup:
            n_miss += 1
            mut_rows.append({
                **base,
                "position_0based": pos0,
                "ddG_pred": None,
                "status": "missing_in_model_output",
            })
            continue

        ddg, wt_from_model = lookup[key]
        status = "ok" if wt_from_model == orig else f"wt_mismatch(model={wt_from_model})"
        if status == "ok":
            ddg_values.append(ddg)
        else:
            n_miss += 1
        mut_rows.append({
            **base,
            "position_0based": pos0,
            "ddG_pred": ddg,
            "status": status,
        })

    ddg_sum = float(sum(ddg_values)) if ddg_values else None
    summary = {
        "winner_idx": winner_idx,
        "year": year,
        "label": f"winner_{winner_idx:02d}",
        "parent": parent,
        "mutations_vs_parent": mut_str,
        "n_mut_total": len(parsed),
        "n_mut_scored": len(ddg_values),
        "n_mut_missing": n_miss,
        "ddG_sum_additive": ddg_sum,
        "ddG_mean": (ddg_sum / len(ddg_values)) if ddg_values else None,
        "ddG_max_single": max(ddg_values) if ddg_values else None,
        "hamming_to_parent": len(parsed),
    }
    return mut_rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", default="avGFP", choices=["avGFP", "sfGFP"],
                    help="相对哪条 WT 求 diff 并查 ThermoMPNN（默认 avGFP）")
    ap.add_argument("--out-winners", default=str(OUTPUTS_DIR / "08d_thermompnn_winners.csv"))
    ap.add_argument("--out-mut", default=str(OUTPUTS_DIR / "08d_thermompnn_winner_mutations.csv"))
    ap.add_argument("--force-rerun", action="store_true",
                    help="忽略已有 ThermoMPNN_inference_*.csv，重跑推理")
    args = ap.parse_args()

    print(f"[1] 读 beforetopseqs from {GFP_DATA_XLSX}")
    win_df = pd.read_excel(GFP_DATA_XLSX, sheet_name="beforetopseqs").reset_index(drop=True)
    win_df["sequence"] = win_df["sequence"].astype(str).str.strip().str.upper()
    print(f"  loaded {len(win_df)} winners; parent={args.parent}")

    wt_map = parse_wt_fasta(WT_FASTA_TXT)
    if args.parent not in wt_map:
        raise KeyError(f"WT FASTA missing {args.parent}")
    wt_seq = wt_map[args.parent]

    print(f"[2] 加载 ThermoMPNN 查表 ({args.parent}) ...")
    csv_path = ensure_inference_csv(args.parent, force=args.force_rerun)
    lookup = load_ddg_lookup(csv_path)

    # 用 WT 单条构造位点映射（08 原逻辑需 seeds.csv）
    wt_stub = pd.DataFrame([{
        "Seq_ID": 0,
        "parent": args.parent,
        "mutations": "",
        "Sequence": wt_seq,
    }])
    with_m_map = build_with_m_to_model_pos_map(wt_stub, args.parent, lookup)
    print(f"  with_M->model map: {len(with_m_map)} positions")

    print("[3] 逐条 winner 查 ddG ...")
    all_mut_rows: list[dict] = []
    all_summaries: list[dict] = []

    for i, r in win_df.iterrows():
        mut_rows, summary = score_winner(
            winner_idx=int(i),
            year=r.get("year", ""),
            parent=args.parent,
            wt_seq=wt_seq,
            win_seq=r["sequence"],
            lookup=lookup,
            with_m_map=with_m_map,
        )
        all_mut_rows.extend(mut_rows)
        all_summaries.append(summary)

    mut_df = pd.DataFrame(all_mut_rows).sort_values(["winner_idx", "position_with_M"])
    win_df_out = pd.DataFrame(all_summaries).sort_values("ddG_sum_additive", ascending=True, na_position="last")
    win_df_out["stability_rank"] = win_df_out["ddG_sum_additive"].rank(
        method="min", ascending=True,
    ).astype("Int64")

    # 可选合并 05c 亮度预测
    bright_path = OUTPUTS_DIR / "05c_winner_predictions.csv"
    if bright_path.is_file():
        bright = pd.read_csv(bright_path)
        bright = bright[~bright["label"].str.startswith("WT_")][
            ["winner_idx", "pred_log10", "ratio_to_avGFP_WT"]
        ]
        win_df_out = win_df_out.merge(bright, on="winner_idx", how="left")

    out_mut = Path(args.out_mut)
    out_win = Path(args.out_winners)
    mut_df.to_csv(out_mut, index=False)
    win_df_out.to_csv(out_win, index=False)

    print(f"\n[done] wrote:\n  - {out_mut}\n  - {out_win}")

    cols = ["label", "year", "n_mut_total", "n_mut_scored", "ddG_sum_additive", "stability_rank"]
    if "ratio_to_avGFP_WT" in win_df_out.columns:
        cols.append("ratio_to_avGFP_WT")
    print("\n[summary] ddG_sum_additive（越负越稳）:")
    print(win_df_out[cols].to_string(index=False))

    ddg = win_df_out["ddG_sum_additive"].dropna()
    n_stable = int((ddg < 0).sum())
    n_unstable = int((ddg > 0).sum())
    print(f"\n[summary] ddG_sum < 0（预测更稳）: {n_stable}/{len(ddg)}  "
          f"ddG_sum > 0（预测 destabilizing）: {n_unstable}/{len(ddg)}")
    if len(ddg):
        print(f"[summary] median ddG_sum={ddg.median():.3f}  "
              f"best={ddg.min():.3f}  worst={ddg.max():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
