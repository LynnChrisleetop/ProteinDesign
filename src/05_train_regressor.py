"""用 ESM 嵌入训练亮度回归模型（替代教程 RF）。

输入：outputs/esm_embeddings.npz
输出：
    outputs/05_model.pkl
    outputs/05_metrics.json
    outputs/05_predictions.csv

策略：
- 默认 LightGBM；--model rf 切回 RandomForest 做基线对比。
- GFP_type 作为分桶特征（amacGFP / avGFP / cgreGFP / ppluGFP），与 ESM 嵌入 concat。
- 80/20 train/test split，固定 random_state=42 以可复现。
- 报 R² / Pearson / Spearman / RMSE / MAE。
- 教程基线 R² = 0.28，我们的目标 ≥ 0.40。

用法：
    python src/05_train_regressor.py
    python src/05_train_regressor.py --model rf --tag esm150m_rf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import OUTPUTS_DIR  # noqa: E402


def load_data(path: Path):
    data = np.load(path, allow_pickle=True)
    return {
        "X": data["embeddings"],
        "y_log10": data["log10_brightness"],
        "y_lin": data["brightness"],
        "gfp_type": data["gfp_type"],
        "mutations": data["mutations"],
        "sequences": data["sequences"],
    }


def make_features(X: np.ndarray, gfp_type: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """concat one-hot GFP type 到 ESM 向量后。"""
    types = sorted(np.unique(gfp_type).tolist())
    onehot = np.zeros((len(gfp_type), len(types)), dtype=np.float32)
    for i, t in enumerate(gfp_type):
        onehot[i, types.index(t)] = 1.0
    feat = np.concatenate([X, onehot], axis=1)
    names = [f"esm_{i}" for i in range(X.shape[1])] + [f"is_{t}" for t in types]
    return feat, names


def fit_lightgbm(Xtr, ytr, Xte, yte, params=None):
    import lightgbm as lgb
    p = dict(
        objective="regression",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=127,
        max_depth=-1,
        min_data_in_leaf=20,
        feature_fraction=0.7,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=42,
        verbose=-1,
    )
    if params:
        p.update(params)
    m = lgb.LGBMRegressor(**p)
    m.fit(
        Xtr, ytr,
        eval_set=[(Xte, yte)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )
    return m


def fit_rf(Xtr, ytr, Xte, yte):
    from sklearn.ensemble import RandomForestRegressor
    m = RandomForestRegressor(
        n_estimators=500, max_depth=None, n_jobs=-1, random_state=42,
        min_samples_leaf=2,
    )
    m.fit(Xtr, ytr)
    return m


def metrics(y_true, y_pred) -> dict:
    return {
        "r2":       float(r2_score(y_true, y_pred)),
        "pearson":  float(pearsonr(y_true, y_pred)[0]),
        "spearman": float(spearmanr(y_true, y_pred)[0]),
        "rmse":     float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":      float(mean_absolute_error(y_true, y_pred)),
        "n":        int(len(y_true)),
    }


def per_gfp_metrics(y_true, y_pred, gfp_types) -> dict:
    out = {}
    for t in sorted(set(gfp_types)):
        mask = gfp_types == t
        if mask.sum() < 5:
            continue
        out[t] = metrics(y_true[mask], y_pred[mask])
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(OUTPUTS_DIR / "esm_embeddings.npz"))
    p.add_argument("--model", default="lgbm", choices=["lgbm", "rf"])
    p.add_argument("--target", default="log10",
                   choices=["log10", "linear"],
                   help="预测 log10 亮度 (推荐) 或 线性亮度")
    p.add_argument("--tag", default="default")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()

    print(f"[1] 读嵌入 {args.data}")
    d = load_data(Path(args.data))
    print(f"  X={d['X'].shape}, y_log10={d['y_log10'].shape}")
    print(f"  gfp_type counts: {pd.Series(d['gfp_type']).value_counts().to_dict()}")

    print(f"[2] 构造特征 (ESM + one-hot GFP)")
    X, feat_names = make_features(d["X"], d["gfp_type"])
    print(f"  X.shape={X.shape}")

    y = d["y_log10"] if args.target == "log10" else d["y_lin"]

    print(f"[3] 80/20 split, stratify by GFP type")
    idx = np.arange(len(X))
    idx_tr, idx_te = train_test_split(
        idx, test_size=args.test_size, random_state=42,
        stratify=d["gfp_type"],
    )
    Xtr, Xte = X[idx_tr], X[idx_te]
    ytr, yte = y[idx_tr], y[idx_te]
    gtr, gte = d["gfp_type"][idx_tr], d["gfp_type"][idx_te]
    print(f"  train={len(Xtr)}  test={len(Xte)}")

    print(f"[4] 训练 ({args.model})")
    if args.model == "lgbm":
        model = fit_lightgbm(Xtr, ytr, Xte, yte)
    else:
        model = fit_rf(Xtr, ytr, Xte, yte)

    print(f"[5] 评估")
    yhat_te = model.predict(Xte)
    yhat_tr = model.predict(Xtr)
    m_te = metrics(yte, yhat_te)
    m_tr = metrics(ytr, yhat_tr)
    m_te_per = per_gfp_metrics(yte, yhat_te, gte)

    print(f"  train: R²={m_tr['r2']:.4f}  pearson={m_tr['pearson']:.4f}")
    print(f"  test : R²={m_te['r2']:.4f}  pearson={m_te['pearson']:.4f}  "
          f"spearman={m_te['spearman']:.4f}  rmse={m_te['rmse']:.4f}")
    print(f"  test per-GFP:")
    for g, mm in m_te_per.items():
        print(f"    {g:>10}  R²={mm['r2']:.4f}  N={mm['n']}")

    if m_te["r2"] >= 0.40:
        print(f"  [GOAL] test R² = {m_te['r2']:.4f} >= 0.40 ✅ 超过教程基线（0.28）")
    elif m_te["r2"] >= 0.28:
        print(f"  [OK] test R² = {m_te['r2']:.4f} 与教程基线持平")
    else:
        print(f"  [WARN] test R² = {m_te['r2']:.4f} < 教程基线，需调参")

    print(f"[6] 保存模型与产物")
    out_dir = OUTPUTS_DIR
    model_path = out_dir / f"05_model_{args.tag}.pkl"
    joblib.dump({
        "model": model,
        "feat_names": feat_names,
        "target": args.target,
        "model_type": args.model,
        "tag": args.tag,
    }, model_path)
    print(f"  -> {model_path}")

    pred_df = pd.DataFrame({
        "split": ["test"] * len(yte),
        "gfp_type": gte,
        "mutations": d["mutations"][idx_te],
        "y_true": yte,
        "y_pred": yhat_te,
        "residual": yte - yhat_te,
    })
    pred_path = out_dir / f"05_predictions_{args.tag}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"  -> {pred_path}")

    summary = {
        "tag": args.tag,
        "model": args.model,
        "target": args.target,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "train": m_tr,
        "test": m_te,
        "test_per_gfp": m_te_per,
    }
    summary_path = out_dir / f"05_metrics_{args.tag}.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {summary_path}")

    print(f"\n[done] 模型已就绪，可在 Day 4 用 model.predict(X_candidate) 筛候选。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
