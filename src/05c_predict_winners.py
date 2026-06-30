"""用 04 嵌入 + 05 模型对 beforetopseqs 20 条往届高分序列做 brightness 推理。

输入：
  - GFP_data.xlsx::beforetopseqs（20 条，2024×10 + 2025×10）
  - outputs/05_model_<tag>.pkl
  - WT FASTA（avGFP / sfGFP baseline）

输出：
  - outputs/05c_winner_predictions.csv（20 winner + 2 WT = 22 行）
  - 终端 summary：median ratio、>=1.0×、>=0.3× 计数

注意：20 条无实验 brightness 标签，仅看 pred_log10 与 ratio_to_avGFP_WT 是否整体偏高。
往届序列更接近 avGFP，GFP type one-hot 默认 parent=avGFP。
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
from utils import GFP_DATA_XLSX, OUTPUTS_DIR, WT_FASTA_TXT, parse_wt_fasta  # noqa: E402


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
    p.add_argument("--model-pkl", default=str(OUTPUTS_DIR / "05_model_esm35m_lgbm.pkl"))
    p.add_argument("--esm-model", default="t12_35M",
                   help="必须与训练 ESM 模型一致")
    p.add_argument("--parent", default="avGFP",
                   help="winner 序列的 GFP type one-hot（默认 avGFP）")
    p.add_argument("--out", default=str(OUTPUTS_DIR / "05c_winner_predictions.csv"))
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

    print(f"[2] 读 beforetopseqs from {GFP_DATA_XLSX}")
    win_df = pd.read_excel(GFP_DATA_XLSX, sheet_name="beforetopseqs")
    win_df = win_df.reset_index(drop=True)
    win_df["sequence"] = win_df["sequence"].astype(str).str.strip().str.upper()
    print(f"  loaded {len(win_df)} winners; parent_onehot={args.parent}")

    wt_map = parse_wt_fasta(WT_FASTA_TXT)
    wt_keys = ["avGFP", "sfGFP"]
    for k in wt_keys:
        if k not in wt_map:
            raise KeyError(f"WT FASTA missing {k}")

    all_seqs: list[str] = []
    all_labels: list[str] = []
    all_years: list[object] = []
    all_widx: list[object] = []
    all_parents: list[str] = []

    for k in wt_keys:
        all_seqs.append(wt_map[k])
        all_labels.append(f"WT_{k}")
        all_years.append("")
        all_widx.append("")
        all_parents.append(k)

    for i, r in win_df.iterrows():
        all_seqs.append(r["sequence"])
        all_labels.append(f"winner_{i:02d}")
        all_years.append(r.get("year", ""))
        all_widx.append(int(i))
        all_parents.append(args.parent)

    print(f"[3] ESM 嵌入 {len(all_seqs)} 条 (WT×{len(wt_keys)} + winners×{len(win_df)}) ...")
    t0 = time.time()
    X_esm = embed_one_batch(all_seqs, args.esm_model, device)
    print(f"  done in {time.time()-t0:.1f}s, X.shape={X_esm.shape}")

    onehot = np.zeros((len(all_seqs), len(gfp_types)), dtype=np.float32)
    for i, par in enumerate(all_parents):
        if par in gfp_types:
            onehot[i, gfp_types.index(par)] = 1.0
    X = np.concatenate([X_esm, onehot], axis=1)
    print(f"  feature.shape={X.shape}  (expected={(len(all_seqs), len(feat_names))})")
    assert X.shape[1] == len(feat_names), "feature width mismatch"

    print("[4] 推断 ...")
    yhat_log10 = model.predict(X)
    yhat_linear = 10 ** yhat_log10

    rows = []
    for lbl, yr, widx, par, seq, yl, ylin in zip(
        all_labels, all_years, all_widx, all_parents, all_seqs, yhat_log10, yhat_linear,
    ):
        rows.append({
            "label": lbl,
            "year": yr,
            "winner_idx": widx,
            "parent_for_onehot": par,
            "length": len(seq),
            "pred_log10": float(yl),
            "pred_linear": float(ylin),
        })
    df = pd.DataFrame(rows)

    wt_lin = {
        r["label"][3:]: r["pred_linear"]
        for _, r in df.iterrows()
        if r["label"].startswith("WT_")
    }
    df["ratio_to_avGFP_WT"] = df["pred_linear"] / wt_lin["avGFP"]
    df["ratio_to_sfGFP_WT"] = df["pred_linear"] / wt_lin["sfGFP"]
    for col in ("ratio_to_avGFP_WT", "ratio_to_sfGFP_WT"):
        df.loc[df["label"].str.startswith("WT_"), col] = 1.0

    df.to_csv(args.out, index=False)
    print(f"\n  -> {args.out}")

    winners = df[~df["label"].str.startswith("WT_")].copy()

    print("\n========== 20 条 winner 预测（按 ratio_to_avGFP_WT 降序）==========")
    print(f"{'label':<14} {'year':<6} {'len':>4}  {'pred_log10':>11}  {'linear':>10}  "
          f"{'ratio_av':>9}  {'ratio_sf':>9}")
    print("-" * 90)
    for _, r in winners.sort_values("ratio_to_avGFP_WT", ascending=False).iterrows():
        print(f"{r['label']:<14} {str(r['year']):<6} {r['length']:>4}  "
              f"{r['pred_log10']:>11.4f}  {r['pred_linear']:>10.1f}  "
              f"{r['ratio_to_avGFP_WT']:>9.3f}  {r['ratio_to_sfGFP_WT']:>9.3f}")

    ratios = winners["ratio_to_avGFP_WT"].values
    n_ge_1 = int((ratios >= 1.0).sum())
    n_ge_03 = int((ratios >= 0.3).sum())
    med = float(np.median(ratios))
    print(f"\n[summary] median ratio_to_avGFP_WT={med:.3f}  "
          f">=1.0×: {n_ge_1}/20  >=0.3×(Finitial 红线): {n_ge_03}/20")
    print("[reminder] 无实验 label，不能与真实 brightness 算 R²；仅检验模型是否整体认可往届设计。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
