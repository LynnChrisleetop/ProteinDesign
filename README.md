# ProteinDesign · 2026 合成生物学创新赛 GFP 设计

> 用计算方法设计 **6 条** GFP 突变序列，最大化「**热处理后亮度相对 WT 的比值**」。
>
> 评分: `S = (Finitial / Finitial_WT) × (Ffinal / Finitial) = Ffinal / Finitial_WT`
>
> 红线: `Finitial < 0.3 × Finitial_WT` → 0 分。 长度 220–250 aa，必须 `M` 开头。

---

## ⏱️ 当前进度（Day 2.5 完成，**6 条种子可提交**）

| 阶段 | 状态 | 产物 | 备注 |
|---|---|---|---|
| **Day 1** · 数据载入 / 合规校验 | ✅ | `outputs/00_summary.json` · `data/processed/wt.fasta` | 5 条 WT 长度全合规（222–238 aa） |
| **Day 2** · 位点池 | ✅ | `outputs/position_pool.csv` · `outputs/lethal_blacklist.csv` 等 | 三股证据合成，致死黑名单 18 个 |
| **Day 2.5** · 6 条种子设计 + 自检 | ✅ | `outputs/seeds.csv` · `outputs/seeds.fasta` · `outputs/07_seed_check.csv` | **6/6 通过**：合规 + 不在 Exclusion + DnaChisel 可合成 |
| **Day 2.5** · 多样性诊断 | ⚠️ | `outputs/03_diversity_report.json` | `NEEDS_DIVERSIFICATION`（仅 2 母本，Seq_2 vs Seq_5 hamming=2） |
| **Day 3** · ESM 嵌入 + 回归模型 | ⏸️ | — | **需 GPU**，下一步：切 V100 |
| **Day 4** · 候选生成（10k+） + 筛选 | ⏸️ | — | 依赖 Day 3 |
| **Day 5** · 热稳定预测（ThermoMPNN） | ⏸️ | — | **需 GPU**，核心评分维度 |
| **Day 6** · 最终 6 条 + `submission.csv` | ✅ 保底就绪 | `outputs/submission.csv` (CRLF, 字节级对齐模板) | 可随时上交保底；ML 阶段产出会覆盖 |
| 设计思路文档 | ✅ | `docs/design_doc.md` | Day 1–2.5 完整决策树 + 关键发现 |
| GPU 启动指南 | ✅ | `docs/gpu_launch.md` | 切 V100/A100 后的逐步开炮指南 |
| agent_log · 公开仓库 | 🔵 | （待写） | 与代码并行整理 |

> ✅ 完成 · ⚠️ 完成但有警告 · ⏸️ 待启动 · 🔵 同步推进

---

## 🧬 当前 6 条种子设计（`outputs/seeds.csv`）

| Seq | 策略 | 母本 | 突变 | 长度 | DNA GC% | 距离最近禁用序列 | 状态 |
|---|---|---|---|---|---|---|---|
| 1 | safe-baseline | avGFP | `S65T:S72A` | 238 | 49.9% | 1 | ✅ |
| 2 | winner-stack | avGFP | `S65T:S72A:Q80R:N105K:V163A` | 238 | 49.9% | 3 | ✅ |
| 3 | sfGFP-control-minus | sfGFP | `S72A` | 238 | 50.7% | 1 | ✅ |
| 4 | boost-engine | avGFP | `F46L:Q157G:V163A` | 238 | 50.3% | 2 | ✅ |
| 5 | gold-rush | avGFP | `F46L:S65T:S72A:Q80R:N105K:Q157G:V163A` | 238 | 50.4% | 5 | ✅ |
| 6 | high-risk-monomer-superboost | sfGFP | `Q157G:A206K` | 238 | 50.6% | 2 | ✅ |

> 全部 6 条经过 `07_dnachisel_check.py`：长度合规、M 开头、20 标准氨基酸、不在 Exclusion_List、DnaChisel 反向翻译成功（E.coli 密码子优化 + 避酶切位点 + GC 30–70%）

---

## 🔬 Day 2 关键发现

### 1. sfGFP = avGFP + 11 个 Superfolder 突变（数据自动确认）

```
S30R, Y39N, S65T, Q80R, F99S, N105T, Y145F, M153T, V163A, I171V, A206V
```

这给了我们"自由切换母本"的灵活性——任何 sfGFP 设计都可表达为 avGFP + 突变包。

### 2. **往届赢家更偏 avGFP 母本**（与教程推荐冲突）

20 条 `beforetopseqs` 历史赢家 vs sfGFP WT 的修改频次（top 5）：

| 位点 | sfGFP 是 | 赢家最常改成 | 频次 | 解读 |
|---|---|---|---|---|
| 145 | F | **Y** | **20/20** | 把 superfolder Y145F **回退** |
| 99 | S | **F** | 19/20 | 把 superfolder F99S **回退** |
| 105 | T | **N** | 15/19 | 把 superfolder N105T **回退** |
| 206 | V | **A** | 18/19 | 把 superfolder A206V **回退** |
| 153 | T | **M** | 18/19 | 把 superfolder M153T **回退** |

→ 历届赢家几乎一致地"还原 avGFP"。**警示**：本届新增 72°C 热稳定考核，sfGFP 的稳定优势可能仍然成立——所以我们保留了 Seq_3/Seq_6 两条 sfGFP 母本做对照。

### 3. **Q157G 是数据驱动的新发现**

avGFP 单点突变 `Q157G` 实测亮度 **2.48× WT**（线性尺度），是 single-point 数据里最强的"增益位"，且**官方教程位点池里没有**。我们把它放进了 Seq_4 / Seq_5 / Seq_6。

### 4. 致死黑名单 18 个（含发色团 Y66，绝对避开）

```
17(E)  19(D)  28(S)  30(S)  32(E)  34(E)  50(T)  56(P)  59(T)
65(S)  66(Y)  84(F)  90(E)  93(V)  95(E)  124(E)  182(Y)  216(D)
```

---

## 🗂️ 仓库结构（**实际**当前状态）

```
ProteinDesign/
├── README.md                    # 这份
├── 参赛指南.md                   # 战略全文（必读）
├── BOHRIUM.md                   # Bohrium 平台操作手册
├── .gitignore
├── scripts/
│   └── bohrium_init.sh          # 一键初始化（装 ESM/ThermoMPNN/ColabDesign 等）
├── src/                         # 流水线脚本（已完成 5 个）
│   ├── utils.py                 # · 路径解析 + FASTA / 突变 / 合规校验
│   ├── 00_load_and_clean.py     # · 数据载入 + WT 校准
│   ├── 01_position_pool.py      # · 三股证据合成位点池（数据+文献+赢家diff）
│   ├── 02_seed_designs.py       # · 6 条种子设计（规则驱动，无 ML）
│   ├── 03_diversity_check.py    # · 6 条之间的 Hamming / 母本多样性
│   ├── 06_make_submission.py    # · 把 seeds.csv 转赛方提交格式（CRLF 对齐）
│   ├── 07_dnachisel_check.py    # · 完整体检：合规 + Exclusion + DnaChisel
│   ├── 04_embed_esm.py          # · ESM2 嵌入（**需 GPU**）
│   ├── 05_train_regressor.py    # · LightGBM 亮度回归（CPU/GPU 均可）
│   # —— 以下待写（依赖 04/05 结果再写）——
│   # ├── 06b_generate_candidates.py
│   # └── 08_thermompnn.py
├── data/
│   ├── raw/                     # （留空，gitignored）赛事数据软链点
│   └── processed/               # 清洗产物（小 CSV / FASTA，进 Git）
│       ├── wt.fasta
│       ├── wt_summary.csv
│       ├── before_top_seqs.csv
│       └── winner_diff_raw.csv
└── outputs/                     # ⭐ 所有报告与候选序列
    ├── 00_summary.json
    ├── 01_summary.json
    ├── 02_summary.json
    ├── 03_diversity_report.json
    ├── 07_summary.json
    ├── seeds.csv                # ⭐ 6 条种子设计（当前最佳）
    ├── seeds.fasta
    ├── submission.csv           # ⭐⭐ 赛方格式（CRLF），可直接提交
    ├── 07_seed_check.csv        # ⭐ 6 条体检详表
    ├── 07_seed_dna.csv          # 6 条的 DNA（DnaChisel 反向翻译）
    ├── position_pool.csv        # 238 位点综合得分
    ├── position_stats_avGFP.csv
    ├── safe_positions.csv       (198 pos)
    ├── boost_positions.csv      (4 pos)
    ├── super_boost_positions.csv (1 pos: Q157G)
    ├── lethal_blacklist.csv     (18 pos)
    ├── literature_priors.csv
    ├── winner_diff_avGFP.csv
    ├── winner_diff_sfGFP.csv
    ├── 03_pairwise_hamming.csv
    ├── 03_dist_to_wt.csv
    └── 03_cgreGFP_candidate.txt
```

赛事数据放在仓库**同级目录** `../2026Protein Design/`（**不进 Git**，详见 [BOHRIUM.md](./BOHRIUM.md)）。

---

## 🚀 复现实验（5 分钟）

### 数据准备

把赛方原始数据放到仓库同级目录：

```
/personal/biosys/
├── 2026Protein Design/      ← 赛方数据（49 MB，含 GFP_data.xlsx 等）
│   ├── GFP_data.xlsx
│   ├── Exclusion_List.csv
│   ├── AAseqs of 5 GFP proteins_20260511.txt
│   ├── submission_template.csv
│   └── referencepaper/...
└── ProteinDesign/           ← 本仓库
```

或在 Bohrium 上勾选挂载 `/bohr/2025proteindesign-iw1n/v1/`（`src/utils.py` 自动识别两种）。

### 环境（CPU 即可跑 Day 1–Day 2.5）

```bash
pip install pandas numpy openpyxl scikit-learn lightgbm biopython joblib \
            matplotlib seaborn dnachisel python-codon-tables
```

GPU 阶段额外：

```bash
pip install torch fair-esm transformers sentencepiece
# 或一键: bash scripts/bohrium_init.sh
```

### 跑流水线

```bash
cd /personal/biosys/ProteinDesign
python src/00_load_and_clean.py     # ~10s   · WT 解析 + 数据基线
python src/01_position_pool.py      # ~10s   · 238 位点综合得分
python src/02_seed_designs.py       # ~1s    · 生成 6 条种子
python src/07_dnachisel_check.py    # ~5s    · 完整体检（含 DnaChisel）
python src/03_diversity_check.py    # ~1s    · 多样性诊断
python src/06_make_submission.py --team <YourTeam> --strict-check
                                    # ~1s    · 生成赛方格式 submission.csv
```

总耗时 < 30 秒。所有产物落到 `outputs/`，无副作用、可复现。

---

## ⏳ 还差什么（按优先级）

### P0 — 提交保底（**已完成**）
- [x] `outputs/submission.csv` 已生成（CRLF + 字节级对齐模板，1575 bytes）
- [x] `docs/design_doc.md`：Day 1–2.5 完整决策树 + 关键发现（→ 后期导出 PDF）
- [x] `docs/gpu_launch.md`：切 GPU 后的逐步开炮指南

### P1 — ML 阶段（**需 GPU，¥5–15**）
- [x] `src/04_embed_esm.py`：ESM2 (35M / 150M / 650M) 嵌入，CRC pool over residues
- [x] `src/05_train_regressor.py`：LightGBM 替代 RF，目标 R² ≥ 0.40（教程仅 0.28）
- [ ] **跑** Day 3.1（ESM 嵌入）+ Day 3.2（训练）→ 拿到 R²
- [ ] `src/06b_generate_candidates.py`（依赖 R² 结果再写）：用 30 位点池生成 10k+ 组合，用模型筛 Top-200
- [ ] 把模型预测最佳的 2–3 条替换当前 Seq_4/5/6 → 重新 `06_make_submission.py`

### P2 — 热稳定预测（**核心评分维度，需 GPU**）
- [ ] `src/08_thermompnn.py`：对所有候选跑 ThermoMPNN 全位点 ΔΔG 扫描
- [ ] 用稳定性预测筛掉「亮度高但 72°C 会炸」的设计
- [ ] 用 ESMFold pLDDT 做结构合理性最终验证（A100）

### P3 — 多样性补强
- [ ] 加 1 条 cgreGFP 母本设计（数据 WT 亮度 31403，比 avGFP 高 6×）— 已草稿在 `outputs/03_cgreGFP_candidate.txt`
- [ ] 加 1 条共识序列设计（多 WT 拼接）

### P4 — 加分项
- [ ] 设计思路 PDF（导出 `docs/design_doc.md`）
- [ ] `docs/agent_log.md`：LLM 决策树记录
- [x] GitHub README 复现说明（本文件）

---

## 🧪 Bohrium 平台速查

| 阶段 | 机型 | 时长 | 成本 |
|---|---|---|---|
| Day 1–2.5（**当前**） | `c4_m8_cpu` | < 1 min/run | ~¥0.3/h |
| Day 3 (ESM 嵌入 + LightGBM) | `1×V100_32g` | ~1.5 h | ~¥8 |
| Day 5 (ThermoMPNN + ESMFold) | `1×A100_40g` | ~2 h | ~¥20 |

详见 [BOHRIUM.md](./BOHRIUM.md)。**记得做完立刻 Stop 实例！**

---

## 📋 文档导航

| 文档 | 用途 |
|---|---|
| **README.md** | 你正在看的这份 — 进度看板 + 复现说明 |
| **[参赛指南.md](./参赛指南.md)** | 战略全文（931 行）：规则解读、6 条梯度策略、位点池、模型管线、提交自检、开源工具箱 |
| **[BOHRIUM.md](./BOHRIUM.md)** | Bohrium 平台操作手册：选镜像、机型、协作、关机省钱 |
| `scripts/bohrium_init.sh` | Bohrium 一键初始化（装环境 + clone 第三方 + 下 PDB） |
| **[docs/design_doc.md](./docs/design_doc.md)** | ⭐ Day 1–2.5 设计思路完整文档，最终导出 PDF 提交 |
| **[docs/gpu_launch.md](./docs/gpu_launch.md)** | ⭐ 切 V100/A100 后逐步开炮指南（含预算预估） |

---

## 复现性

- 所有 Day 1–2.5 脚本在 Bohrium `ubuntu:22.04-py3.10-cuda12.1` 镜像 + Python 3.10 + CPU 4 核 + 8 GB 内存下测试通过
- 第三方工具版本（ProteinMPNN、ThermoMPNN、ColabDesign、ESM-3）通过 `scripts/bohrium_init.sh` 锁定
- 比赛数据集（`GFP_data.xlsx`、`Exclusion_List.csv` 等）**未在仓库中重新分发**，遵守赛方"不外泄"约定

---

## License

Code: MIT · Documentation: CC BY-SA 4.0

第三方工具各遵循其原协议（详见 [参赛指南.md](./参赛指南.md) §14）。

---

## 团队

LynnChrisleetop · 2026 合成生物学创新赛
