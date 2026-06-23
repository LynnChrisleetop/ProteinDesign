# ProteinDesign · 2026 合成生物学创新赛 GFP 设计

> 用计算方法设计 **6 条** GFP 突变序列，最大化「**热处理后亮度相对 WT 的比值**」。
>
> 评分: `S = (Finitial / Finitial_WT) × (Ffinal / Finitial) = Ffinal / Finitial_WT`
>
> 红线: `Finitial < 0.3 × Finitial_WT` → 0 分。 长度 220–250 aa，必须 `M` 开头。
>
> **队名：新次元小队**

---

## ⏱️ 当前进度（Day 3.4 完成，**submission.csv 最终版已就绪**）

| 阶段 | 状态 | 产物 | 备注 |
|---|---|---|---|
| **Day 1** · 数据载入 / 合规校验 | ✅ | `outputs/00_summary.json` · `data/processed/wt.fasta` | 5 条 WT 长度全合规（222–238 aa） |
| **Day 2** · 位点池 | ✅ | `outputs/position_pool.csv` · `outputs/lethal_blacklist.csv` 等 | 三股证据合成，致死黑名单 18 个 |
| **Day 2.5** · 6 条种子 + 合规体检 | ✅ | `outputs/seeds.csv` · `outputs/07_seed_check.csv` | 6/6 通过：合规 + Exclusion + DnaChisel |
| **Day 2.5** · 多样性诊断 | ✅ | `outputs/03_diversity_report.json` | GOOD（pairwise min Hamming = 3） |
| **Day 3.1** · ESM2-35M 嵌入 | ✅ | `data/processed/esm_embeddings.npz` (141,365 序列) | V100 耗时 9.57 min，`(N,480)` 矩阵 |
| **Day 3.2** · LightGBM 亮度回归 | ✅ | `outputs/model_lgbm.pkl` · `outputs/05_eval.json` | **Test R²=0.714**（基线 0.28，2.5×提升） |
| **Day 3.3** · 种子亮度预测 | ✅ | `outputs/05b_seed_predictions.csv` | Seq_1/3/6 比值 0.6–0.9，Seq_2/4/5 见备注 |
| **Day 3.3b** · ML 候选生成与筛选 | ✅ | `outputs/06b_top_candidates.csv`（Top-20） | 5000 组合扫描，**最优 ratio=1.27** |
| **Day 3.4** · 用 ML Top-1 替换 Seq_5 | ✅ | `outputs/submission.csv` ⭐ | 最终版，队名=新次元小队，CRLF 字节级对齐 |
| **Day 5** · 热稳定预测（ThermoMPNN） | ⏸️ | — | 可选加分项，需 GPU |
| 设计思路文档 | ✅ | `docs/design_doc.md` | Day 1–2.5 完整决策树 + 关键发现 |
| GPU 启动指南 | ✅ | `docs/gpu_launch.md` | Bohrium V100/A100 逐步开炮指南 |

> ✅ 完成 · ⚠️ 完成但有警告 · ⏸️ 待启动

---

## 🧬 最终 6 条设计（`outputs/seeds.csv`，已生成 `submission.csv`）

| Seq | 策略 | 母本 | 突变 | 长度 | DNA GC% | 距禁用序列 | ML ratio | 状态 |
|---|---|---|---|---|---|---|---|---|
| 1 | safe-baseline | avGFP | `S65T:S72A` | 238 | 49.6% | 1 | 0.88× | ✅ |
| 2 | winner-stack | avGFP | `S65T:S72A:Q80R:N105K:V163A` | 238 | 50.0% | 3 | — | ✅ |
| 3 | sfGFP-control | sfGFP | `S72A` | 238 | 50.7% | 1 | — | ✅ |
| 4 | boost-engine | avGFP | `F46L:K158G:V163A` | 238 | 50.6% | 2 | — | ✅ |
| **5** | **ML-top1** | **avGFP** | **`S65T:S72A:N105Y:S147N:I171S:L178V`** | 238 | 49.7% | **5** | **1.27×** ⭐ | ✅ |
| 6 | high-risk-monomer | sfGFP | `K158G:A206K` | 238 | 50.8% | 2 | — | ✅ |

> **Seq_5** 是 ESM+LightGBM 从 5000 个候选组合中筛出的最优序列（ratio=1.27），包含赢家高频 tolerant 位点 N105Y + S147N + I171S + L178V，与现有种子最小 Hamming = 5。
>
> Seq_2/4/6 的 ML ratio 偏低（0.03–0.07）是因为模型训练数据为室温初始亮度，而这些设计侧重热稳定，与竞赛评分维度不完全吻合；历史赢家在同一模型上也呈现类似低估。

> v1.1 修正：Seq_4/5/6 原写 `Q157G` 是 Sarkisyan 数据集 skip-M 编号与 with-M 编号混用所致误读；
> 真实 super-boost 是 **`K158G`**（avGFP `WT[157]=K`）。详见 [docs/design_doc.md](./docs/design_doc.md) §3.6。

---

## 🔬 Day 2 关键发现

### 1. sfGFP = avGFP + 11 个 Superfolder 突变（数据自动确认）

```
S30R, Y39N, S65T, Q80R, F99S, N105T, Y145F, M153T, V163A, I171V, A206V
```

这给了我们"自由切换母本"的灵活性——任何 sfGFP 设计都可表达为 avGFP + 突变包。

### 2. 往届赢家更偏 avGFP 母本（与教程推荐冲突）

20 条 `beforetopseqs` 历史赢家 vs sfGFP WT 的修改频次（top 5）：

| 位点 | sfGFP 是 | 赢家最常改成 | 频次 | 解读 |
|---|---|---|---|---|
| 145 | F | **Y** | **20/20** | 把 superfolder Y145F **回退** |
| 99 | S | **F** | 19/20 | 把 superfolder F99S **回退** |
| 105 | T | **N** | 15/19 | 把 superfolder N105T **回退** |
| 206 | V | **A** | 18/19 | 把 superfolder A206V **回退** |
| 153 | T | **M** | 18/19 | 把 superfolder M153T **回退** |

→ 历届赢家几乎一致地"还原 avGFP"。本届新增 72°C 热稳定考核，sfGFP 稳定优势可能成立——所以保留了 Seq_3/6 两条 sfGFP 母本做对照。

### 3. K158G 是数据驱动的新发现（v1.1 修正：原 Q157G 误读）

avGFP 单点突变 `K158G`（with-M 编号）实测亮度 **2.48× WT**（线性尺度），是 single-point 数据里最强的增益位，且**官方教程位点池里没有**。放进了 Seq_4 / Seq_6。

> **编号体系坑**：Sarkisyan 数据集用 skip-M 编号（起始 M 不算 1）；文献/赛方 FASTA/赢家序列用 with-M 编号（M=1）。两套位号差 1。`utils.detect_numbering()` + `strict=True` 双保险防止再犯。

### 4. 致死黑名单 18 个（含发色团 Y66，绝对避开）

```
17(E)  19(D)  28(S)  30(S)  32(E)  34(E)  50(T)  56(P)  59(T)
65(S)  66(Y)  84(F)  90(E)  93(V)  95(E)  124(E)  182(Y)  216(D)
```

### 5. ML 模型训练结果（Day 3）

- **ESM2-35M** 嵌入 141,365 条序列，`(N, 480)` 矩阵，耗时 9.57 min（V100）
- **LightGBM** 回归：Test R²=**0.714**，Pearson=0.845，Spearman=0.858（基线 RF R²=0.28）
- 候选生成：5000 个随机组合 → ESM 嵌入 → 模型打分 → 多样性过滤 → **Top-20 候选**
- 最优候选 `S65T:S72A:N105Y:S147N:I171S:L178V` 预测 ratio=**1.27×**，已纳入 Seq_5

---

## 🗂️ 仓库结构

```
ProteinDesign/
├── README.md                    # 进度看板 + 复现说明
├── 参赛指南.md                   # 战略全文（必读）
├── BOHRIUM.md                   # Bohrium 平台操作手册
├── .gitignore
├── scripts/
│   └── bohrium_init.sh          # 一键初始化环境
├── src/                         # 流水线脚本（全部完成）
│   ├── utils.py                 # 路径 / FASTA / 突变 / 合规校验（含编号系统处理）
│   ├── 00_load_and_clean.py     # 数据载入 + WT 校准
│   ├── 01_position_pool.py      # 三股证据合成位点池
│   ├── 02_seed_designs.py       # 6 条种子（规则 + ML 结果）
│   ├── 03_diversity_check.py    # Hamming / 母本多样性诊断
│   ├── 04_embed_esm.py          # ESM2 嵌入（需 GPU）
│   ├── 05_train_regressor.py    # LightGBM 亮度回归
│   ├── 05b_predict_seeds.py     # 用训练好模型预测 6 条种子
│   ├── 06_make_submission.py    # 生成赛方格式 submission.csv（CRLF）
│   ├── 06b_generate_candidates.py  # ML 候选生成 + 多样性筛选
│   └── 07_dnachisel_check.py    # 完整体检（合规 + Exclusion + DnaChisel）
├── data/
│   ├── raw/                     # （gitignored）赛事数据
│   └── processed/               # 清洗产物（进 Git）
│       ├── wt.fasta
│       ├── wt_summary.csv
│       ├── before_top_seqs.csv
│       └── winner_diff_raw.csv
├── docs/
│   ├── design_doc.md            # ⭐ 完整设计思路文档（导出 PDF 提交）
│   └── gpu_launch.md            # Bohrium GPU 逐步指南 + 预算
└── outputs/                     # 所有报告与候选序列
    ├── submission.csv           # ⭐⭐ 最终提交文件（CRLF，字节级对齐模板）
    ├── seeds.csv / seeds.fasta  # ⭐ 6 条种子当前版本
    ├── 07_seed_check.csv        # 6 条体检详表（合规 + Hamming + DNA）
    ├── 06b_top_candidates.csv   # ML 筛选 Top-20 候选
    ├── 06b_candidates_all.csv   # ML 筛选全量候选（5000 条）
    ├── model_lgbm.pkl           # 训练好的 LightGBM 模型
    ├── position_pool.csv        # 238 位点综合得分
    ├── lethal_blacklist.csv     # 致死黑名单（18 位点）
    └── ...（其他中间产物）
```

赛事数据放在仓库**同级目录** `../2026Protein Design/`（**不进 Git**，详见 [BOHRIUM.md](./BOHRIUM.md)）。

---

## 🚀 复现实验

### 数据准备

```
/personal/biosys/
├── 2026Protein Design/      ← 赛方数据（含 GFP_data.xlsx 等）
│   ├── GFP_data.xlsx
│   ├── Exclusion_List.csv
│   ├── AAseqs of 5 GFP proteins_20260511.txt
│   └── submission_template.csv
└── ProteinDesign/           ← 本仓库
```

或在 Bohrium 上勾选挂载 `/bohr/2025proteindesign-iw1n/v1/`（`src/utils.py` 自动识别）。

### 环境

```bash
# CPU 阶段（Day 1–2.5）
pip install pandas numpy openpyxl scikit-learn lightgbm biopython joblib \
            matplotlib seaborn dnachisel python-codon-tables

# GPU 阶段额外（Day 3+）
pip install torch fair-esm transformers sentencepiece
```

### 跑流水线

```bash
cd /personal/biosys/ProteinDesign

# Day 1–2.5（CPU，<30s）
python src/00_load_and_clean.py
python src/01_position_pool.py
python src/02_seed_designs.py
python src/07_dnachisel_check.py
python src/03_diversity_check.py
python src/06_make_submission.py --team 新次元小队 --strict-check

# Day 3（GPU）
python src/04_embed_esm.py                      # ~10 min on V100
python src/05_train_regressor.py                # ~2 min
python src/05b_predict_seeds.py                 # <1 min
python src/06b_generate_candidates.py --n-samples 5000 --top-k 20
```

---

## ⏳ 还差什么（按优先级）

### P0 — 提交就绪（**已完成** ✅）
- [x] `outputs/submission.csv`：最终版，队名=新次元小队，CRLF + 字节级对齐模板
- [x] `docs/design_doc.md`：Day 1–2.5 完整决策树（后续导出 PDF）
- [x] Seq_5 已替换为 ML 筛选最优候选（ratio=1.27）

### P1 — ML 阶段（**已完成** ✅）
- [x] ESM2-35M 嵌入 141k 序列（Test R²=0.714）
- [x] LightGBM 训练，显著超越教程基线
- [x] 候选生成 5000 条，筛出 Top-20
- [x] ML 最优候选纳入最终 submission

### P2 — 热稳定预测（可选，¥10–20）
- [ ] `src/08_thermompnn.py`：ThermoMPNN 全位点 ΔΔG 扫描
- [ ] 用稳定性筛掉「亮但 72°C 会炸」的设计
- [ ] ESMFold pLDDT 结构合理性验证

### P3 — 多样性补强（可选）
- [ ] 加 1 条 cgreGFP 母本设计（数据 WT 亮度 31403，比 avGFP 高 6×）
- [ ] 草稿：`outputs/03_cgreGFP_candidate.txt`

### P4 — 加分项
- [ ] `docs/design_doc.md` 导出为 PDF
- [ ] `docs/agent_log.md`：LLM 决策树记录

---

## 🧪 Bohrium 平台速查

| 阶段 | 机型 | 时长 | 成本 |
|---|---|---|---|
| Day 1–2.5（CPU） | `c4_m8_cpu` | < 1 min/run | ~¥0.3/h |
| Day 3（ESM 嵌入 + LightGBM） | `1×V100_32g` | ~15 min | ~¥1.5 |
| Day 5（ThermoMPNN + ESMFold） | `1×A100_40g` | ~2 h | ~¥20 |

详见 [BOHRIUM.md](./BOHRIUM.md)。**记得做完立刻 Stop 实例！**

---

## 📋 文档导航

| 文档 | 用途 |
|---|---|
| **README.md** | 进度看板 + 复现说明（本文件） |
| **[参赛指南.md](./参赛指南.md)** | 战略全文：规则解读、梯度策略、位点池、模型管线、提交自检 |
| **[BOHRIUM.md](./BOHRIUM.md)** | Bohrium 平台操作手册 |
| **[docs/design_doc.md](./docs/design_doc.md)** | ⭐ 完整设计思路文档，最终导出 PDF 提交 |
| **[docs/gpu_launch.md](./docs/gpu_launch.md)** | ⭐ GPU 实例逐步开炮指南（含预算） |

---

## 复现性

- Day 1–2.5 脚本在 Bohrium `ubuntu:22.04-py3.10-cuda12.1` + Python 3.10 + CPU 4 核 8 GB 下测试通过
- Day 3 脚本在 V100 32GB / Tesla T4 15GB 下测试通过
- 比赛数据集（`GFP_data.xlsx`、`Exclusion_List.csv` 等）**未在仓库中重新分发**，遵守赛方约定

---

## License

Code: MIT · Documentation: CC BY-SA 4.0

---

## 团队

**新次元小队** · LynnChrisleetop · 2026 合成生物学创新赛
