"""ESM2 嵌入 brightness 表中的所有 4 类 GFP 突变体序列。

输入：GFP_data.xlsx::brightness（5 列：aaMutations / brightness / log10brightness / GFP_type / WT_aaseq）
输出：
    outputs/esm_embeddings.npz   {ids, sequences, embeddings(N, D), targets(N,), gfp_type(N,)}
    outputs/04_summary.json

设计要点：
- 模型默认 esm2_t12_35M_UR50D（最快，11 MB），命令行 --model 可换 t30_150M / t33_650M。
- 训练数据是 5–141k 行；ESM 嵌入只跑一次缓存复用。
- 对每条序列做 mean-pool over residues（去掉 BOS/EOS）。

用法：
    # 35M (最快, V100 5 min, 适合验证管线)
    python src/04_embed_esm.py --model t12_35M
    # 150M (V100 20 min, 推荐产出)
    python src/04_embed_esm.py --model t30_150M --batch 16
    # 650M (V100 90 min+, 顶配)
    python src/04_embed_esm.py --model t33_650M --batch 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    GFP_DATA_XLSX,
    OUTPUTS_DIR,
    apply_mutations,
    parse_wt_fasta,
    WT_FASTA_TXT,
)

MODEL_MAP = {
    "t6_8M":    ("esm2_t6_8M_UR50D",   320),
    "t12_35M":  ("esm2_t12_35M_UR50D", 480),
    "t30_150M": ("esm2_t30_150M_UR50D", 640),
    "t33_650M": ("esm2_t33_650M_UR50D", 1280),
}


def build_sequences(wt_map: dict[str, str], df: pd.DataFrame) -> pd.DataFrame:
    """把 (aaMutations, GFP_type) 还原成完整 AA 序列。

    aaMutations 为空 → WT 本体。
    """
    rows = []
    skipped = 0
    for i, r in df.iterrows():
        gfp = str(r["GFP_type"]).strip()
        mut = "" if pd.isna(r["aaMutations"]) else str(r["aaMutations"]).strip()
        wt = wt_map.get(gfp)
        if wt is None:
            skipped += 1
            continue
        if mut == "":
            seq = wt
        else:
            seq = apply_mutations(wt, mut)
            if seq is None:
                skipped += 1
                continue
        rows.append({
            "row_idx": int(i),
            "gfp_type": gfp,
            "mutations": mut,
            "log10_brightness": float(r["log10brightness"]),
            "brightness": float(r["brightness"]) if "brightness" in r else np.nan,
            "sequence": seq,
        })
    print(f"  built {len(rows)} sequences (skipped {skipped})")
    return pd.DataFrame(rows)


def embed_sequences(
    seqs: list[str],
    model_key: str,
    batch_size: int,
    device: str,
    max_len: int = 1024,
) -> np.ndarray:
    import torch
    import esm

    model_name, embed_dim = MODEL_MAP[model_key]
    print(f"[embed] loading {model_name} (embed_dim={embed_dim}) on {device}")
    model_ctor = getattr(esm.pretrained, model_name)
    model, alphabet = model_ctor()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device).eval()
    n_layers = model.num_layers

    out = np.zeros((len(seqs), embed_dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i:i + batch_size]
            data = [(f"s{i+j}", s[:max_len]) for j, s in enumerate(chunk)]
            _, _, toks = batch_converter(data)
            toks = toks.to(device)
            results = model(toks, repr_layers=[n_layers], return_contacts=False)
            reps = results["representations"][n_layers]  # (B, L, D)
            for j, (_, s) in enumerate(data):
                L = len(s)
                emb = reps[j, 1:L + 1].mean(dim=0).cpu().numpy()
                out[i + j] = emb
            if (i // batch_size) % 20 == 0:
                dt = time.time() - t0
                rate = (i + len(chunk)) / max(dt, 0.001)
                eta = (len(seqs) - i - len(chunk)) / max(rate, 0.001)
                print(f"  [{i + len(chunk):>6}/{len(seqs)}] "
                      f"{rate:.1f} seq/s, ETA {eta/60:.1f} min")
    print(f"[done] embedded {len(seqs)} in {(time.time()-t0)/60:.2f} min")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="t12_35M",
                   choices=list(MODEL_MAP.keys()),
                   help="ESM2 model size (default t12_35M)")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--limit", type=int, default=0,
                   help="只取前 N 行（0 = 全跑）")
    p.add_argument("--gfp-only", default="",
                   help="只跑某类（avGFP/amacGFP/cgreGFP/ppluGFP），逗号分隔；空=全跑")
    p.add_argument("--out", default=str(OUTPUTS_DIR / "esm_embeddings.npz"))
    args = p.parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    if device == "cpu":
        print("[WARN] 没有 GPU，这一步在 CPU 上会非常慢（35M ≈ 3 h / 全集）")
        print("       仅用于本地烟测；正式跑请切 V100/A100。")

    print(f"[1] 读 brightness 表 ({GFP_DATA_XLSX})")
    df = pd.read_excel(GFP_DATA_XLSX, sheet_name="brightness")
    print(f"  {len(df)} rows, columns={df.columns.tolist()}")

    wt_map = parse_wt_fasta(WT_FASTA_TXT)
    print(f"  WT map: {list(wt_map.keys())}")

    if args.gfp_only:
        only = {x.strip() for x in args.gfp_only.split(",") if x.strip()}
        df = df[df["GFP_type"].isin(only)].reset_index(drop=True)
        print(f"  filter GFP_type={only} -> {len(df)} rows")

    if args.limit > 0:
        df = df.head(args.limit).copy()
        print(f"  --limit {args.limit} -> {len(df)} rows")

    print(f"[2] 还原完整氨基酸序列 ...")
    seqs_df = build_sequences(wt_map, df)

    seqs_df = seqs_df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)
    print(f"  去重后 {len(seqs_df)} 条")

    print(f"[3] ESM2 嵌入 (model={args.model}, batch={args.batch}) ...")
    embeddings = embed_sequences(
        seqs_df["sequence"].tolist(),
        args.model,
        args.batch,
        device,
    )

    print(f"[4] 保存到 {args.out}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        log10_brightness=seqs_df["log10_brightness"].values,
        brightness=seqs_df["brightness"].values,
        gfp_type=seqs_df["gfp_type"].astype(str).values,
        mutations=seqs_df["mutations"].astype(str).values,
        sequences=seqs_df["sequence"].astype(str).values,
    )
    size_mb = Path(args.out).stat().st_size / (1024 * 1024)
    print(f"  -> {size_mb:.1f} MB")

    summary = {
        "model": args.model,
        "device": device,
        "n_sequences": int(len(seqs_df)),
        "embed_dim": int(embeddings.shape[1]),
        "gfp_type_counts": seqs_df["gfp_type"].value_counts().to_dict(),
        "out_file": args.out,
        "out_size_mb": round(size_mb, 1),
    }
    with (OUTPUTS_DIR / "04_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] {json.dumps(summary, indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
