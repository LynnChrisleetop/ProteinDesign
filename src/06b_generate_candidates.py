"""Day 3.3b  候选序列组合搜索 + ML 排序

策略：
  Tier-A 骨架（必选）: S65T + S72A（EGFP 经典 + winner 最高频）
  Tier-B 随机扩展  : 从 position_pool.csv Top-25 非致死位点中随机取 1-4 个
  母本              : avGFP (70%) + sfGFP (30%)

筛选流程：
  1. 合规（length/M/AA/Exclusion）
  2. ESM2 嵌入 + LightGBM 预测 log10 brightness
  3. 与现有 6 条种子以及本批 Top 候选 Hamming ≥ 3 多样性过滤
  4. 输出 Top-K 推荐（用于替换 Seq_5/Seq_4 等高风险槽位）

用法:
  python src/06b_generate_candidates.py              # 默认 5000 样本, Top-20
  python src/06b_generate_candidates.py --n-samples 10000 --top-k 30
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    EXCLUSION_CSV,
    OUTPUTS_DIR,
    WT_FASTA_TXT,
    apply_mutations,
    check_sequence,
    parse_wt_fasta,
)

TIER_A = ["S65T", "S72A"]   # 必选骨架
MIN_HAMMING = 3              # 与任何已有种子的最小 Hamming 距离


def hamming(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b))
    return sum(x != y for x, y in zip(a, b))


def build_tier_b(pool: pd.DataFrame, lethal_set: set[int],
                 skip_pos: set[int], top_n: int = 25) -> list[tuple[int, str, str]]:
    out: list[tuple[int, str, str]] = []
    for _, r in pool.sort_values("score", ascending=False).iterrows():
        p = int(r["pos"])
        if p in lethal_set or p in skip_pos:
            continue
        if r["score"] <= 0:
            continue
        wt_aa = str(r["wt_aa_avGFP"])
        new_aa = str(r.get("best_substitution", ""))
        if not new_aa or new_aa in (".", "*", "nan") or wt_aa == new_aa:
            continue
        out.append((p, wt_aa, new_aa))
        if len(out) >= top_n:
            break
    return out


def generate(av: str, sf: str, tier_b: list[tuple[int, str, str]],
             n: int, av_frac: float = 0.7, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    tier_a_parsed = [(m[0], int(m[1:-1]), m[-1]) for m in TIER_A]
    tier_a_pos = {p for _, p, _ in tier_a_parsed}

    for _ in range(n * 6):
        if len(out) >= n:
            break
        parent = "avGFP" if rng.random() < av_frac else "sfGFP"
        wt = av if parent == "avGFP" else sf
        k = rng.choice([1, 1, 2, 2, 3, 4])
        k = min(k, len(tier_b))
        extra = rng.sample(tier_b, k)
        all_muts: dict[int, tuple[str, str]] = {p: (o, n) for _, p, n in tier_a_parsed
                                                 for o in [wt[p - 1]]}
        for p, o, n in extra:
            all_muts[p] = (o, n)
        mut_str = ":".join(f"{o}{p}{n}" for p, (o, n) in sorted(all_muts.items()))
        key = (parent, mut_str)
        if key in seen:
            continue
        seen.add(key)
        seq = apply_mutations(wt, mut_str)
        if seq is None:
            continue
        out.append({"parent": parent, "mutations": mut_str,
                    "n_mut": len(all_muts), "sequence": seq})
    return out


def embed_batched(seqs: list[str], model_key: str, device: str,
                  batch: int = 64) -> np.ndarray:
    import torch
    import esm

    MODEL_MAP = {
        "t6_8M":    ("esm2_t6_8M_UR50D",    320),
        "t12_35M":  ("esm2_t12_35M_UR50D",  480),
        "t30_150M": ("esm2_t30_150M_UR50D", 640),
    }
    name, _ = MODEL_MAP[model_key]
    print(f"  loading {name} ...")
    m, alph = getattr(esm.pretrained, name)()
    bc = alph.get_batch_converter()
    m = m.to(device).eval()
    n_layers = m.num_layers

    out: list[np.ndarray] = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            chunk = seqs[i:i + batch]
            data = [(f"s{i+j}", s) for j, s in enumerate(chunk)]
            _, _, toks = bc(data)
            toks = toks.to(device)
            res = m(toks, repr_layers=[n_layers], return_contacts=False)
            reps = res["representations"][n_layers]
            for j, (_, s) in enumerate(data):
                out.append(reps[j, 1:len(s) + 1].mean(0).cpu().numpy())
            if i % (batch * 10) == 0:
                dt = time.time() - t0
                rate = (i + len(chunk)) / max(dt, 1e-3)
                eta = (len(seqs) - i - len(chunk)) / max(rate, 1e-3)
                print(f"  [{i+len(chunk):>5}/{len(seqs)}]  {rate:.0f} seq/s  ETA {eta/60:.1f} min")
    print(f"  done in {(time.time()-t0)/60:.2f} min")
    return np.stack(out, axis=0).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--esm-model", default="t12_35M")
    ap.add_argument("--model-pkl",
                    default=str(OUTPUTS_DIR / "05_model_esm35m_lgbm.pkl"))
    ap.add_argument("--pool",   default=str(OUTPUTS_DIR / "position_pool.csv"))
    ap.add_argument("--lethal", default=str(OUTPUTS_DIR / "lethal_blacklist.csv"))
    ap.add_argument("--seeds",  default=str(OUTPUTS_DIR / "seeds.csv"))
    ap.add_argument("--rng-seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ── 读取基础数据 ──────────────────────────────────────────────
    print("\n[1] 读取 WT / pool / lethal / seeds / exclusion ...")
    wt = parse_wt_fasta(WT_FASTA_TXT)
    av, sf = wt["avGFP"], wt["sfGFP"]
    pool    = pd.read_csv(args.pool)
    lethal  = pd.read_csv(args.lethal)
    lethal_set = {int(r["pos"]) for _, r in lethal.iterrows()}
    seeds   = pd.read_csv(args.seeds).dropna(subset=["Sequence"])
    seed_seqs = [str(s).strip().upper() for s in seeds["Sequence"]]
    excl    = pd.read_csv(EXCLUSION_CSV)
    excl_set = {str(s).strip().upper()
                for s in excl[excl.columns[0]].tolist()}
    print(f"  pool={len(pool)}  lethal={len(lethal_set)}  "
          f"seeds={len(seeds)}  excl={len(excl_set)}")

    tier_a_pos = {int(m[1:-1]) for m in TIER_A}
    tier_b = build_tier_b(pool, lethal_set, tier_a_pos, top_n=25)
    print(f"\n[2] Tier-B pool ({len(tier_b)} positions):")
    for p, o, n in tier_b:
        print(f"    {o}{p}{n}")

    # ── 生成候选 ──────────────────────────────────────────────────
    print(f"\n[3] 随机采样 {args.n_samples} 个候选 ...")
    cands = generate(av, sf, tier_b, args.n_samples, seed=args.rng_seed)
    print(f"  generated {len(cands)} unique")

    keep = [c for c in cands
            if check_sequence(c["sequence"])["passes_all"]
            and c["sequence"].upper() not in excl_set]
    print(f"  {len(keep)} pass compliance + exclusion")
    if not keep:
        print("[FATAL] 没有合规候选，扩大 --n-samples")
        return 1

    # ── ESM 嵌入 + 推断 ───────────────────────────────────────────
    print(f"\n[4] ESM 嵌入 ({args.esm_model}, {len(keep)} seqs) ...")
    X_esm = embed_batched([c["sequence"] for c in keep],
                          args.esm_model, device)

    print(f"\n[5] 加载 LightGBM 模型，推断 ...")
    mdl = joblib.load(args.model_pkl)
    model     = mdl["model"]
    feat_names = mdl["feat_names"]
    gfp_types  = [n[3:] for n in feat_names if n.startswith("is_")]

    oh = np.zeros((len(keep), len(gfp_types)), dtype=np.float32)
    for i, c in enumerate(keep):
        if c["parent"] in gfp_types:
            oh[i, gfp_types.index(c["parent"])] = 1.0
    X   = np.concatenate([X_esm, oh], axis=1)
    yh  = model.predict(X)

    # WT baseline
    def wt_pred(seq: str, parent: str) -> float:
        X_wt = embed_batched([seq], args.esm_model, device, batch=1)
        oh_wt = np.zeros((1, len(gfp_types)), dtype=np.float32)
        if parent in gfp_types:
            oh_wt[0, gfp_types.index(parent)] = 1.0
        return float(model.predict(np.concatenate([X_wt, oh_wt], axis=1))[0])

    av_y = wt_pred(av, "avGFP")
    sf_y = wt_pred(sf, "avGFP")   # sfGFP 没训练数据，用 avGFP one-hot 代理
    print(f"  avGFP WT pred log10={av_y:.4f}  "
          f"sfGFP WT pred log10={sf_y:.4f} (avGFP proxy)")

    # ── 收集结果 ─────────────────────────────────────────────────
    rows = []
    for i, c in enumerate(keep):
        wt_log = av_y if c["parent"] == "avGFP" else sf_y
        ratio  = 10 ** (yh[i] - wt_log)
        min_h  = min(hamming(c["sequence"], s) for s in seed_seqs)
        rows.append({
            **c,
            "pred_log10": float(yh[i]),
            "pred_linear": float(10 ** yh[i]),
            "ratio_to_parent_WT": float(ratio),
            "min_hamming_to_seeds": int(min_h),
        })

    df = (pd.DataFrame(rows)
          .sort_values("pred_log10", ascending=False)
          .reset_index(drop=True))

    # 保存全量（不含序列，省空间）
    df.drop(columns=["sequence"]).to_csv(
        OUTPUTS_DIR / "06b_candidates_all.csv", index=False)
    print(f"\n  -> 06b_candidates_all.csv  ({len(df)} rows)")

    # ── 多样性 Top-K ─────────────────────────────────────────────
    print(f"\n[6] Top-{args.top_k} 多样性过滤 (Hamming ≥ {MIN_HAMMING}) ...")
    chosen: list[dict] = []
    chosen_seqs = list(seed_seqs)
    for _, r in df.iterrows():
        s = r["sequence"]
        if min(hamming(s, x) for x in chosen_seqs) < MIN_HAMMING:
            continue
        chosen.append(r.to_dict())
        chosen_seqs.append(s)
        if len(chosen) >= args.top_k:
            break

    top_df = pd.DataFrame(chosen)
    top_df.drop(columns=["sequence"]).to_csv(
        OUTPUTS_DIR / "06b_top_candidates.csv", index=False)
    print(f"  -> 06b_top_candidates.csv  ({len(top_df)} rows)")

    # ── 打印摘要 ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"Top {min(15, len(top_df))} 推荐（按预测 log10 亮度降序）")
    print(f"{'parent':<8} {'n_mut':>5} {'log10':>7} {'ratio':>7} "
          f"{'h_seeds':>7}  mutations")
    print("-" * 80)
    for _, r in top_df.head(15).iterrows():
        print(f"{r['parent']:<8} {int(r['n_mut']):>5} "
              f"{r['pred_log10']:>7.4f} {r['ratio_to_parent_WT']:>7.3f} "
              f"{int(r['min_hamming_to_seeds']):>7}  {r['mutations']}")

    summary = {
        "esm_model": args.esm_model,
        "n_sampled": args.n_samples,
        "n_unique": len(cands),
        "n_compliant": len(keep),
        "n_top": len(top_df),
        "best_pred_log10": float(top_df["pred_log10"].iloc[0]) if len(top_df) else None,
        "best_ratio": float(top_df["ratio_to_parent_WT"].iloc[0]) if len(top_df) else None,
        "best_mutations": str(top_df["mutations"].iloc[0]) if len(top_df) else None,
        "av_y": av_y,
        "sf_y_proxy": sf_y,
    }
    (OUTPUTS_DIR / "06b_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] -> outputs/06b_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
