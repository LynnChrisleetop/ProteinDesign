"""Day 4: 仅对 6 条 seeds 做 ThermoMPNN 热稳定评分。

输出:
  - outputs/08_thermompnn_mutations.csv  每个突变位点的 ddG
  - outputs/08_thermompnn_seeds.csv      每条 seed 的汇总评分

说明:
  - ThermoMPNN 默认输出的是单点突变 ddG。
  - 对多突变序列，这里用 "单点 ddG 求和" 作为近似总分（越负越稳）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from Bio import pairwise2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS_DIR, parse_mutation_str  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
THERMOMPNN_SCRIPT = REPO_ROOT / "third_party" / "ThermoMPNN" / "analysis" / "custom_inference.py"
PDB_DIR = REPO_ROOT / "inputs" / "pdb"

PARENT_TO_PDB = {
    "avGFP": ("2WUR", "A"),
    "sfGFP": ("2B3P", "A"),
}


def ensure_inference_csv(parent: str, force: bool = False) -> Path:
    if parent not in PARENT_TO_PDB:
        raise ValueError(f"unsupported parent: {parent!r}")
    pdb_id, chain = PARENT_TO_PDB[parent]
    out_csv = OUTPUTS_DIR / f"ThermoMPNN_inference_{pdb_id}.csv"
    if out_csv.is_file() and not force:
        return out_csv

    pdb_path = PDB_DIR / f"{pdb_id}.pdb"
    if not pdb_path.is_file():
        raise FileNotFoundError(f"missing pdb: {pdb_path}")

    cmd = [
        sys.executable,
        str(THERMOMPNN_SCRIPT),
        "--pdb",
        str(pdb_path),
        "--chain",
        chain,
        "--out_dir",
        str(OUTPUTS_DIR),
    ]
    print(f"[run] ThermoMPNN {parent} ({pdb_id}, chain {chain}) ...")
    subprocess.run(cmd, check=True)
    if not out_csv.is_file():
        raise FileNotFoundError(f"inference csv not generated: {out_csv}")
    return out_csv


def load_ddg_lookup(csv_path: Path) -> dict[tuple[int, str], tuple[float, str]]:
    df = pd.read_csv(csv_path)
    df["position"] = df["position"].astype(int)
    lookup: dict[tuple[int, str], tuple[float, str]] = {}
    for _, r in df.iterrows():
        key = (int(r["position"]), str(r["mutation"]))
        lookup[key] = (float(r["ddG_pred"]), str(r["wildtype"]))
    return lookup


def reconstruct_parent_sequence(seeds: pd.DataFrame, parent: str) -> str:
    """用某条 seed 序列 + mutation 反推母本序列。"""
    sub = seeds[seeds["parent"] == parent].copy()
    if sub.empty:
        raise ValueError(f"no seeds for parent={parent!r}")
    # 优先用突变最少的那条，反推误差最低
    sub["n_mut"] = sub["mutations"].astype(str).map(lambda s: len(parse_mutation_str(s) or []))
    row = sub.sort_values("n_mut").iloc[0]
    seq = list(str(row["Sequence"]).strip())
    parsed = parse_mutation_str(str(row["mutations"]).strip()) or []
    for orig, pos_with_m, _new in parsed:
        idx = pos_with_m - 1
        if 0 <= idx < len(seq):
            seq[idx] = orig
    return "".join(seq)


def build_with_m_to_model_pos_map(
    seeds: pd.DataFrame,
    parent: str,
    lookup: dict[tuple[int, str], tuple[float, str]],
) -> dict[int, int]:
    """把 with_M 1-based 位号映射到 ThermoMPNN position（处理缺失残基）。"""
    parent_seq = reconstruct_parent_sequence(seeds, parent)
    wt_by_pos: dict[int, str] = {}
    for (pos0, _mut), (_ddg, wt) in lookup.items():
        wt_by_pos[pos0] = wt

    positions = sorted(wt_by_pos)
    model_seq = "".join(wt_by_pos[p] for p in positions)
    with_m_to_model: dict[int, int] = {}

    # 全局对齐，稳健处理缺失残基与轻微错位
    aln = pairwise2.align.globalms(parent_seq, model_seq, 2.0, -1.0, -5.0, -0.2, one_alignment_only=True)
    if not aln:
        return with_m_to_model
    parent_aln, model_aln, _score, _start, _end = aln[0]

    i = 0  # parent_seq 0-based
    j = 0  # model_seq 0-based
    for a, b in zip(parent_aln, model_aln):
        if a != "-":
            i += 1
        if b != "-":
            j += 1
        if a != "-" and b != "-":
            with_m_to_model[i] = positions[j - 1]
    return with_m_to_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default=str(OUTPUTS_DIR / "seeds.csv"))
    ap.add_argument("--out-seed", default=str(OUTPUTS_DIR / "08_thermompnn_seeds.csv"))
    ap.add_argument("--out-mut", default=str(OUTPUTS_DIR / "08_thermompnn_mutations.csv"))
    ap.add_argument("--force-rerun", action="store_true", help="忽略已有 ThermoMPNN_inference_*.csv")
    args = ap.parse_args()

    seeds = pd.read_csv(args.seeds).dropna(subset=["Sequence", "mutations", "parent"])
    seeds = seeds.sort_values("Seq_ID").reset_index(drop=True)

    lookups: dict[str, dict[tuple[int, str], tuple[float, str]]] = {}
    with_m_maps: dict[str, dict[int, int]] = {}
    for parent in sorted(seeds["parent"].unique()):
        csv_path = ensure_inference_csv(parent, force=args.force_rerun)
        lookups[parent] = load_ddg_lookup(csv_path)
        with_m_maps[parent] = build_with_m_to_model_pos_map(seeds, parent, lookups[parent])
        print(f"[info] built with_M->model map for {parent}: {len(with_m_maps[parent])} positions")

    mut_rows: list[dict] = []
    seed_rows: list[dict] = []

    for _, r in seeds.iterrows():
        seq_id = int(r["Seq_ID"])
        parent = str(r["parent"])
        mut_str = str(r["mutations"]).strip()
        parsed = parse_mutation_str(mut_str) or []
        lookup = lookups[parent]
        with_m_map = with_m_maps[parent]

        ddg_values: list[float] = []
        n_miss = 0
        for orig, pos_with_m, new in parsed:
            if pos_with_m not in with_m_map:
                n_miss += 1
                mut_rows.append(
                    {
                        "Seq_ID": seq_id,
                        "parent": parent,
                        "mutation": f"{orig}{pos_with_m}{new}",
                        "position_with_M": pos_with_m,
                        "position_0based": None,
                        "ddG_pred": None,
                        "status": "position_not_covered_in_pdb",
                    }
                )
                continue

            pos0 = with_m_map[pos_with_m]
            key = (pos0, new)
            if key not in lookup:
                n_miss += 1
                mut_rows.append(
                    {
                        "Seq_ID": seq_id,
                        "parent": parent,
                        "mutation": f"{orig}{pos_with_m}{new}",
                        "position_with_M": pos_with_m,
                        "position_0based": pos0,
                        "ddG_pred": None,
                        "status": "missing_in_model_output",
                    }
                )
                continue

            ddg, wt_from_model = lookup[key]
            status = "ok" if wt_from_model == orig else f"wt_mismatch(model={wt_from_model})"
            if status.startswith("ok"):
                ddg_values.append(ddg)
            else:
                n_miss += 1
            mut_rows.append(
                {
                    "Seq_ID": seq_id,
                    "parent": parent,
                    "mutation": f"{orig}{pos_with_m}{new}",
                    "position_with_M": pos_with_m,
                    "position_0based": pos0,
                    "ddG_pred": ddg,
                    "status": status,
                }
            )

        ddg_sum = float(sum(ddg_values)) if ddg_values else None
        ddg_mean = (ddg_sum / len(ddg_values)) if ddg_values else None
        ddg_max = max(ddg_values) if ddg_values else None
        seed_rows.append(
            {
                "Seq_ID": seq_id,
                "strategy": r.get("strategy", ""),
                "parent": parent,
                "mutations": mut_str,
                "n_mut_total": len(parsed),
                "n_mut_scored": len(ddg_values),
                "n_mut_missing": n_miss,
                "ddG_sum_additive": ddg_sum,
                "ddG_mean": ddg_mean,
                "ddG_max_single": ddg_max,
            }
        )

    mut_df = pd.DataFrame(mut_rows).sort_values(["Seq_ID", "position_with_M"])
    seed_df = pd.DataFrame(seed_rows).sort_values("ddG_sum_additive", ascending=True, na_position="last")
    if "ddG_sum_additive" in seed_df.columns:
        seed_df["stability_rank"] = seed_df["ddG_sum_additive"].rank(method="min", ascending=True).astype("Int64")

    out_mut = Path(args.out_mut)
    out_seed = Path(args.out_seed)
    mut_df.to_csv(out_mut, index=False)
    seed_df.to_csv(out_seed, index=False)

    print(f"\n[done] wrote:\n  - {out_mut}\n  - {out_seed}")
    print("\n[summary] ddG_sum_additive (越负越稳):")
    print(seed_df[["Seq_ID", "parent", "mutations", "ddG_sum_additive", "stability_rank"]].to_string(index=False))


if __name__ == "__main__":
    main()
