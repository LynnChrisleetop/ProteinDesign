"""Day 4: 对 06b 候选前 N 条做 ThermoMPNN 热稳定打分并给出替换建议。

默认输入:
  - outputs/06b_candidates_all.csv
  - outputs/seeds.csv

默认输出:
  - outputs/08c_top200_mutations.csv
  - outputs/08c_top200_scored.csv
  - outputs/08c_replacement_suggestions.csv
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
    return out_csv


def load_ddg_lookup(csv_path: Path) -> dict[tuple[int, str], tuple[float, str]]:
    df = pd.read_csv(csv_path)
    lookup: dict[tuple[int, str], tuple[float, str]] = {}
    for _, r in df.iterrows():
        lookup[(int(r["position"]), str(r["mutation"]))] = (float(r["ddG_pred"]), str(r["wildtype"]))
    return lookup


def reconstruct_parent_sequence(seeds: pd.DataFrame, parent: str) -> str:
    sub = seeds[seeds["parent"] == parent].copy()
    if sub.empty:
        raise ValueError(f"no seed sequence for parent={parent!r}")
    sub["n_mut"] = sub["mutations"].astype(str).map(lambda s: len(parse_mutation_str(s) or []))
    row = sub.sort_values("n_mut").iloc[0]
    seq = list(str(row["Sequence"]).strip())
    for orig, pos_with_m, _new in (parse_mutation_str(str(row["mutations"])) or []):
        idx = pos_with_m - 1
        if 0 <= idx < len(seq):
            seq[idx] = orig
    return "".join(seq)


def build_with_m_to_model_pos_map(
    seeds: pd.DataFrame, parent: str, lookup: dict[tuple[int, str], tuple[float, str]]
) -> dict[int, int]:
    parent_seq = reconstruct_parent_sequence(seeds, parent)
    wt_by_pos: dict[int, str] = {}
    for (pos0, _mut), (_ddg, wt) in lookup.items():
        wt_by_pos[pos0] = wt
    positions = sorted(wt_by_pos)
    model_seq = "".join(wt_by_pos[p] for p in positions)
    out: dict[int, int] = {}
    aln = pairwise2.align.globalms(parent_seq, model_seq, 2.0, -1.0, -5.0, -0.2, one_alignment_only=True)
    if not aln:
        return out
    parent_aln, model_aln, _score, _start, _end = aln[0]
    i = 0
    j = 0
    for a, b in zip(parent_aln, model_aln):
        if a != "-":
            i += 1
        if b != "-":
            j += 1
        if a != "-" and b != "-":
            out[i] = positions[j - 1]
    return out


def score_mutations(
    parent: str,
    mut_str: str,
    lookup: dict[tuple[int, str], tuple[float, str]],
    with_m_map: dict[int, int],
) -> tuple[list[dict], dict]:
    parsed = parse_mutation_str(mut_str) or []
    mut_rows: list[dict] = []
    ddg_values: list[float] = []
    n_missing = 0
    n_mismatch = 0
    for orig, pos_with_m, new in parsed:
        if pos_with_m not in with_m_map:
            n_missing += 1
            mut_rows.append(
                {
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
            n_missing += 1
            mut_rows.append(
                {
                    "mutation": f"{orig}{pos_with_m}{new}",
                    "position_with_M": pos_with_m,
                    "position_0based": pos0,
                    "ddG_pred": None,
                    "status": "missing_in_model_output",
                }
            )
            continue
        ddg, wt = lookup[key]
        status = "ok" if wt == orig else f"wt_mismatch(model={wt})"
        if status == "ok":
            ddg_values.append(ddg)
        else:
            n_mismatch += 1
        mut_rows.append(
            {
                "mutation": f"{orig}{pos_with_m}{new}",
                "position_with_M": pos_with_m,
                "position_0based": pos0,
                "ddG_pred": ddg,
                "status": status,
            }
        )

    summary = {
        "n_mut_total": len(parsed),
        "n_mut_scored_ok": len(ddg_values),
        "n_mut_missing": n_missing,
        "n_mut_mismatch": n_mismatch,
        "ddG_sum_ok": float(sum(ddg_values)) if ddg_values else None,
        "ddG_mean_ok": (float(sum(ddg_values)) / len(ddg_values)) if ddg_values else None,
    }
    return mut_rows, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(OUTPUTS_DIR / "06b_candidates_all.csv"))
    ap.add_argument("--seeds", default=str(OUTPUTS_DIR / "seeds.csv"))
    ap.add_argument("--top-n", type=int, default=200)
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--out-mut", default=str(OUTPUTS_DIR / "08c_top200_mutations.csv"))
    ap.add_argument("--out-scored", default=str(OUTPUTS_DIR / "08c_top200_scored.csv"))
    ap.add_argument("--out-suggest", default=str(OUTPUTS_DIR / "08c_replacement_suggestions.csv"))
    args = ap.parse_args()

    seeds = pd.read_csv(args.seeds).dropna(subset=["Sequence", "mutations", "parent"])
    cands = pd.read_csv(args.candidates).dropna(subset=["parent", "mutations"])
    cands = cands.sort_values("pred_log10", ascending=False).head(args.top_n).reset_index(drop=True)

    lookups: dict[str, dict[tuple[int, str], tuple[float, str]]] = {}
    with_m_maps: dict[str, dict[int, int]] = {}
    for parent in sorted(set(cands["parent"].unique()) | set(seeds["parent"].unique())):
        if parent not in PARENT_TO_PDB:
            continue
        csv_path = ensure_inference_csv(parent, force=args.force_rerun)
        lookups[parent] = load_ddg_lookup(csv_path)
        with_m_maps[parent] = build_with_m_to_model_pos_map(seeds, parent, lookups[parent])
        print(f"[info] map {parent}: {len(with_m_maps[parent])} positions")

    mut_rows = []
    scored_rows = []
    for idx, r in cands.iterrows():
        parent = str(r["parent"])
        mut_str = str(r["mutations"])
        if parent not in lookups:
            continue
        each_mut, summary = score_mutations(parent, mut_str, lookups[parent], with_m_maps[parent])
        cand_id = idx + 1
        for m in each_mut:
            mut_rows.append({"candidate_id": cand_id, "parent": parent, "mutations": mut_str, **m})
        scored_rows.append(
            {
                "candidate_id": cand_id,
                "parent": parent,
                "mutations": mut_str,
                "pred_log10": float(r["pred_log10"]),
                "ratio_to_parent_WT": float(r["ratio_to_parent_WT"]),
                "min_hamming_to_seeds": int(r["min_hamming_to_seeds"]),
                **summary,
            }
        )

    mut_df = pd.DataFrame(mut_rows)
    scored_df = pd.DataFrame(scored_rows)
    scored_df["stability_rank"] = scored_df["ddG_sum_ok"].rank(method="min", ascending=True).astype("Int64")
    scored_df["brightness_rank"] = scored_df["pred_log10"].rank(method="min", ascending=False).astype("Int64")
    scored_df["combo_rank"] = (0.5 * scored_df["stability_rank"] + 0.5 * scored_df["brightness_rank"]).rank(
        method="min", ascending=True
    ).astype("Int64")
    scored_df = scored_df.sort_values("combo_rank")

    # 用当前 6 条作为基线，给替换建议（优先替换热稳定最差的三条）
    seeds_scored = pd.read_csv(OUTPUTS_DIR / "08_thermompnn_seeds.csv")
    weak = seeds_scored.sort_values("ddG_sum_additive", ascending=False).head(3)
    suggest = scored_df[
        (scored_df["n_mut_missing"] <= 1)
        & (scored_df["n_mut_mismatch"] <= 1)
        & (scored_df["min_hamming_to_seeds"] >= 3)
    ].head(15)
    suggest_rows = []
    for _, w in weak.iterrows():
        for _, s in suggest.head(3).iterrows():
            suggest_rows.append(
                {
                    "replace_seq_id": int(w["Seq_ID"]),
                    "replace_mutations": str(w["mutations"]),
                    "replace_ddG_sum": float(w["ddG_sum_additive"]) if pd.notna(w["ddG_sum_additive"]) else None,
                    "candidate_id": int(s["candidate_id"]),
                    "candidate_parent": str(s["parent"]),
                    "candidate_mutations": str(s["mutations"]),
                    "candidate_ddG_sum": float(s["ddG_sum_ok"]) if pd.notna(s["ddG_sum_ok"]) else None,
                    "candidate_ratio": float(s["ratio_to_parent_WT"]),
                    "candidate_combo_rank": int(s["combo_rank"]),
                }
            )
    suggest_df = pd.DataFrame(suggest_rows)

    Path(args.out_mut).parent.mkdir(parents=True, exist_ok=True)
    mut_df.to_csv(args.out_mut, index=False)
    scored_df.to_csv(args.out_scored, index=False)
    suggest_df.to_csv(args.out_suggest, index=False)

    print(f"\n[done] wrote:\n  - {args.out_mut}\n  - {args.out_scored}\n  - {args.out_suggest}")
    print("\n[top 10 by combo_rank]")
    print(
        scored_df[
            ["candidate_id", "parent", "mutations", "ratio_to_parent_WT", "ddG_sum_ok", "combo_rank"]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()
