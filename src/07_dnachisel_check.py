"""Day 2.5 自检：对 outputs/seeds.csv 6 条种子做完整体检。

检查项：
  A. 基础合规     · 长度 220-250 / M 开头 / 20 标准氨基酸
  B. Exclusion    · exact-match  +  最近邻 Hamming 距离（同长度，numpy 加速）
  C. DnaChisel    · AA→DNA 反向翻译，E. coli 密码子优化，验证可合成
                     约束：GC 30-70%, 避 EcoRI/BamHI/HindIII/NdeI, 无连续 6 个相同碱基

输出：
  - outputs/07_seed_check.csv     · 每条种子的详细体检报告
  - outputs/07_seed_dna.csv       · DnaChisel 反向翻译后的 DNA 序列
  - outputs/07_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    DATA_DIR,
    EXCLUSION_CSV,
    OUTPUTS_DIR,
    check_sequence,
)


def nearest_hamming(seed: str, by_len: dict[int, np.ndarray]) -> dict:
    """返回 seed 到 Exclusion_List 中【同长度】序列的最小 Hamming。

    by_len: {seq_len: char_matrix (N, L)} numpy 'S1' 矩阵
    """
    L = len(seed)
    if L not in by_len:
        return {"min_hamming": None, "n_same_length": 0, "closest_idx": None}
    seed_arr = np.frombuffer(seed.encode("ascii"), dtype="S1")
    mat = by_len[L]
    diff = (mat != seed_arr).sum(axis=1)
    idx = int(np.argmin(diff))
    return {"min_hamming": int(diff[idx]), "n_same_length": int(mat.shape[0]),
            "closest_idx": idx}


def dnachisel_check(aa_seq: str) -> dict:
    """对 AA 序列做反向翻译 + 合成可行性优化。"""
    try:
        from dnachisel import (
            DnaOptimizationProblem,
            CodonOptimize,
            EnforceTranslation,
            EnforceGCContent,
            AvoidPattern,
            reverse_translate,
        )
    except Exception as e:
        return {"ok": False, "reason": f"dnachisel import failed: {e}"}

    t0 = time.time()
    try:
        dna = reverse_translate(aa_seq)
    except Exception as e:
        return {"ok": False, "reason": f"reverse_translate failed: {e}"}

    try:
        problem = DnaOptimizationProblem(
            sequence=dna,
            constraints=[
                EnforceTranslation(),
                EnforceGCContent(mini=0.30, maxi=0.70, window=80),
                AvoidPattern("GAATTC"),   # EcoRI
                AvoidPattern("GGATCC"),   # BamHI
                AvoidPattern("AAGCTT"),   # HindIII
                AvoidPattern("CATATG"),   # NdeI
                AvoidPattern("AAAAAA"),
                AvoidPattern("TTTTTT"),
                AvoidPattern("GGGGGG"),
                AvoidPattern("CCCCCC"),
            ],
            objectives=[CodonOptimize(species="e_coli")],
            logger=None,
        )
        problem.resolve_constraints()
        problem.optimize()
        ok = problem.all_constraints_pass()
        final_dna = problem.sequence
        gc = (final_dna.count("G") + final_dna.count("C")) / len(final_dna)
        elapsed = time.time() - t0
        return {
            "ok": bool(ok),
            "elapsed_s": round(elapsed, 2),
            "dna_length": len(final_dna),
            "gc_content": round(gc, 4),
            "constraints_pass": bool(ok),
            "dna_sequence": final_dna,
        }
    except Exception as e:
        return {"ok": False, "reason": f"dnachisel optimize failed: {e}",
                "elapsed_s": round(time.time() - t0, 2)}


def main() -> int:
    if DATA_DIR is None:
        print("[FATAL] no data dir", file=sys.stderr)
        return 1

    print(f"[info] DATA_DIR = {DATA_DIR}")

    print("\n[1] 读 outputs/seeds.csv ...")
    seeds = pd.read_csv(OUTPUTS_DIR / "seeds.csv")
    seeds = seeds.dropna(subset=["Sequence"]).reset_index(drop=True)
    print(f"  loaded {len(seeds)} seeds")

    print("\n[2] 读 Exclusion_List.csv 并按长度索引 ...")
    excl_df = pd.read_csv(EXCLUSION_CSV)
    col = excl_df.columns[0]
    excl_seqs = excl_df[col].astype(str).str.strip().str.upper().tolist()
    excl_set = set(excl_seqs)

    by_len: dict[int, np.ndarray] = {}
    for L, group in pd.Series(excl_seqs).groupby(pd.Series(excl_seqs).str.len()):
        if not 200 <= L <= 260:
            continue
        seqs = group.tolist()
        try:
            mat = np.array([np.frombuffer(s.encode("ascii"), dtype="S1") for s in seqs])
            by_len[L] = mat
        except Exception:
            pass
    print(f"  按长度索引 (200-260 范围)：")
    for L in sorted(by_len.keys()):
        print(f"    len={L:3d}  n={by_len[L].shape[0]}")

    print("\n[3] 逐条体检 ...")
    rows = []
    dna_rows = []
    for _, r in seeds.iterrows():
        sid = int(r["Seq_ID"])
        seq = str(r["Sequence"]).strip().upper()
        print(f"\n  --- Seq_{sid} ({r['strategy']}) ---")
        print(f"  parent={r['parent']}  mutations={r['mutations']}  len={len(seq)}")

        chk = check_sequence(seq)
        in_excl = seq in excl_set

        nn = nearest_hamming(seq, by_len)
        if nn["min_hamming"] is not None:
            print(f"  Exclusion 同长度 Hamming 最近 = {nn['min_hamming']} "
                  f"(在 {nn['n_same_length']} 条 len={len(seq)} 序列中)")
            if nn["min_hamming"] <= 3:
                print(f"  ⚠️  距离禁用序列仅 {nn['min_hamming']} 步，可能与历史提交太像")

        print(f"  DnaChisel 反向翻译中 ...")
        dna_res = dnachisel_check(seq)
        if dna_res["ok"]:
            print(f"  [ok] DNA len={dna_res['dna_length']}  GC={dna_res['gc_content']*100:.1f}%  "
                  f"耗时 {dna_res['elapsed_s']}s  约束全过")
        else:
            print(f"  [FAIL] {dna_res.get('reason', 'unknown')}")

        rows.append({
            "Seq_ID": sid,
            "strategy": r["strategy"],
            "parent": r["parent"],
            "mutations": r["mutations"],
            "length": chk["length"],
            "starts_with_M": chk["starts_with_M"],
            "only_standard_aa": chk["only_standard_aa"],
            "basic_passes": chk["passes_all"],
            "in_exclusion": in_excl,
            "nearest_excl_hamming": nn["min_hamming"],
            "n_same_length_excl": nn["n_same_length"],
            "dna_ok": dna_res["ok"],
            "dna_gc": dna_res.get("gc_content"),
            "dna_elapsed_s": dna_res.get("elapsed_s"),
            "dna_fail_reason": None if dna_res["ok"] else dna_res.get("reason"),
            "verdict": (
                "OK" if (chk["passes_all"] and not in_excl and dna_res["ok"])
                else ("REJECT" if in_excl or not chk["passes_all"] else "REVIEW")
            ),
            "note": (
                f"hamming={nn['min_hamming']} (与禁用序列距离极近，但合规)"
                if nn["min_hamming"] is not None and nn["min_hamming"] <= 1
                else ""
            ),
        })
        if dna_res.get("dna_sequence"):
            dna_rows.append({
                "Seq_ID": sid,
                "dna_sequence": dna_res["dna_sequence"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUTS_DIR / "07_seed_check.csv", index=False)
    pd.DataFrame(dna_rows).to_csv(OUTPUTS_DIR / "07_seed_dna.csv", index=False)
    print(f"\n  -> outputs/07_seed_check.csv")
    print(f"  -> outputs/07_seed_dna.csv ({len(dna_rows)} sequences)")

    print("\n[4] 汇总报告 ...")
    print(out[["Seq_ID", "strategy", "length", "in_exclusion",
               "nearest_excl_hamming", "dna_ok", "dna_gc", "verdict", "note"]].to_string(index=False))

    n_ok = int((out["verdict"] == "OK").sum())
    n_review = int((out["verdict"] == "REVIEW").sum())
    n_reject = int((out["verdict"] == "REJECT").sum())

    summary = {
        "n_seeds": len(out),
        "n_OK": n_ok,
        "n_REVIEW": n_review,
        "n_REJECT": n_reject,
        "exclusion_list_size": len(excl_set),
        "lengths_indexed": {int(L): int(by_len[L].shape[0]) for L in by_len},
        "seed_report": out.to_dict(orient="records"),
    }
    (OUTPUTS_DIR / "07_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n[done] OK={n_ok}/{len(out)}  REVIEW={n_review}  REJECT={n_reject}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
