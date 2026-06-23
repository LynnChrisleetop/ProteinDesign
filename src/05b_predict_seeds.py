"""用 04 嵌入 + 05 模型给当前 6 条种子打 brightness 分。

输入：
  - outputs/seeds.csv (6 条 + Sequence)
  - outputs/05_model_<tag>.pkl
  - WT FASTA（用于算 WT baseline 预测）

输出：
  - outputs/05b_seed_predictions.csv
  - 终端表格：每条预测 log10 brightness、相对 av/sfGFP WT 的 ratio

注意：ML 模型预测的是 *initial* brightness（没考虑 72°C 热处理），
仅用于评估 Finitial >= 0.3 × WT 这一红线、以及"亮度增益是否合理"。
真正比赛得分 = Ffinal / WT，要看热稳定预测（Day 4）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS_DIR, WT_FASTA_TXT, parse_wt_fasta  # noqa: E402


def embed_one_batch(seqs: list[str], model_key: str, device: str) -> np.ndarray:
    import torch
    import esm
    MODEL_MAP = {
        "t6_8M":    ("esm2_t6_8M_UR50D",   320),
        "t12_35M":  ("esm2_t12_35M_UR50D", 480),
        "t30_150M": ("esm2_t30_150M_UR50D", 640),
        "t33_650M": ("esm2_t33_650M_UR50D", 1280),
    }
    model_name, _ = MODEL_MAP[model_key]
    model_ctor = getattr(esm.pretrained, model_name)
    model, alphabet = model_ctor()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    n_layers = model.num_layers
    with torch.no_grad():
        data = [(f"s{i}", s) for i, s in enumerate(seqs)]
        _, _, toks = batch_converter(data)
        toks = toks.to(device)
        results = model(toks, repr_layers=[n_layers], return_contacts=False)
        reps = results["representations"][n_layers]
        out = []
        for j, (_, s) in enumerate(data):
            emb = reps[j, 1:len(s) + 1].mean(dim=0).cpu().numpy()
            out.append(emb)
    return np.stack(out, axis=0).astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default=str(OUTPUTS_DIR / "seeds.csv"))
    p.add_argument("--model-pkl", default=str(OUTPUTS_DIR / "05_model_esm35m_lgbm.pkl"))
    p.add_argument("--esm-model", default="t12_35M",
                   help="必须与训练 ESM 模型一致")
    p.add_argument("--out", default=str(OUTPUTS_DIR / "05b_seed_predictions.csv"))
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    print(f"[1] 加载训练好的模型 {args.model_pkl}")
    mdl_pack = joblib.load(args.model_pkl)
    model = mdl_pack["model"]
    feat_names = mdl_pack["feat_names"]
    target = mdl_pack["target"]
    embed_dim = sum(1 for n in feat_names if n.startswith("esm_"))
    gfp_types = [n[3:] for n in feat_names if n.startswith("is_")]
    print(f"  target={target}  embed_dim={embed_dim}  gfp_types={gfp_types}")

    print(f"[2] 读种子 {args.seeds}")
    seeds = pd.read_csv(args.seeds).dropna(subset=["Sequence"])
    seeds = seeds.sort_values("Seq_ID").reset_index(drop=True)

    wt_map = parse_wt_fasta(WT_FASTA_TXT)
    wt_records = [(k, v) for k, v in wt_map.items()]

    all_seqs = []
    all_labels = []
    all_parents = []
    for k, v in wt_records:
        all_seqs.append(v)
        all_labels.append(f"WT_{k}")
        all_parents.append(k)
    for _, r in seeds.iterrows():
        all_seqs.append(str(r["Sequence"]).strip().upper())
        all_labels.append(f"Seq_{int(r['Seq_ID'])}({r['strategy']})")
        all_parents.append(str(r["parent"]).strip())

    print(f"[3] ESM 嵌入 {len(all_seqs)} 条 (WT×{len(wt_records)} + seeds×{len(seeds)}) ...")
    t0 = time.time()
    X_esm = embed_one_batch(all_seqs, args.esm_model, device)
    print(f"  done in {time.time()-t0:.1f}s, X.shape={X_esm.shape}")

    onehot = np.zeros((len(all_seqs), len(gfp_types)), dtype=np.float32)
    for i, par in enumerate(all_parents):
        if par in gfp_types:
            onehot[i, gfp_types.index(par)] = 1.0
        else:
            pass
    X = np.concatenate([X_esm, onehot], axis=1)
    print(f"  feature.shape={X.shape}  (expected={(len(all_seqs), len(feat_names))})")
    assert X.shape[1] == len(feat_names), "feature width mismatch"

    print(f"[4] 推断 ...")
    yhat_log10 = model.predict(X)
    yhat_linear = 10 ** yhat_log10

    rows = []
    for i, (lbl, par, seq, yl, ylin) in enumerate(zip(
        all_labels, all_parents, all_seqs, yhat_log10, yhat_linear,
    )):
        rows.append({
            "i": i, "label": lbl, "parent": par, "length": len(seq),
            "pred_log10": float(yl),
            "pred_linear": float(ylin),
        })
    df = pd.DataFrame(rows)

    wt_lin = {r["label"][3:]: r["pred_linear"] for _, r in df.iterrows()
              if r["label"].startswith("WT_")}

    def ratio(row):
        par = row["parent"]
        if par in wt_lin and not row["label"].startswith("WT_"):
            return row["pred_linear"] / wt_lin[par]
        return np.nan
    df["ratio_to_parent_WT"] = df.apply(ratio, axis=1)

    df.to_csv(args.out, index=False)
    print(f"\n  -> {args.out}")

    print("\n========== 预测结果 ==========")
    print(f"{'label':<48} {'parent':<8} {'len':>4}  {'pred_log10':>11}  {'linear':>10}  {'ratio_to_WT':>11}")
    print("-" * 110)
    for _, r in df.iterrows():
        rt = f"{r['ratio_to_parent_WT']:.3f}" if not np.isnan(r["ratio_to_parent_WT"]) else "—"
        print(f"{r['label']:<48} {r['parent']:<8} {r['length']:>4}  "
              f"{r['pred_log10']:>11.4f}  {r['pred_linear']:>10.1f}  {rt:>11}")

    print()
    seeds_df = df[~df["label"].str.startswith("WT_")].copy()
    print(f"[ratio_to_parent_WT 排序]")
    for _, r in seeds_df.sort_values("ratio_to_parent_WT", ascending=False).iterrows():
        marker = ""
        if r["ratio_to_parent_WT"] >= 1.5: marker = " 🚀"
        elif r["ratio_to_parent_WT"] >= 1.0: marker = " ✓"
        elif r["ratio_to_parent_WT"] >= 0.3: marker = " ~"
        else: marker = " ❌ (红线!)"
        print(f"  {r['label']:<48}  ratio={r['ratio_to_parent_WT']:.3f}{marker}")
    print()
    print("[reminder] 这是 *initial* brightness 预测，没考虑 72°C 热稳定（核心评分维度待 Day 4）。")
    print("[reminder] ratio >= 0.3 是 Finitial 红线最低保活；ratio >= 1.0 是与 WT 持平；>= 1.5 是有意义增益。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
