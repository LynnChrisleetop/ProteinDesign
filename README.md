# ProteinDesign · 2026 合成生物学创新赛 GFP 蛋白设计

> **队伍**：新次元小队  
> **开源仓库**：https://github.com/LynnChrisleetop/ProteinDesign  
> **任务**：设计 6 条 GFP 突变蛋白序列，最大化 72°C 热处理后相对 WT 的亮度比值。

---

## 赛题与评分（简述）

\[
S = \frac{F_{\text{final}}}{F_{\text{initial}}^{\text{WT}}}
\]

硬约束：长度 220–250 aa、`M` 开头、20 种标准氨基酸、6 条互不相同、不得与 `Exclusion_List.csv` 中任意序列 **完全一致**。

**最终提交文件**：`outputs/submission.csv`（已通过 Exclusion + DnaChisel + 赛方模板自检）。

---

## 流水线总览

本仓库按 **阶段** 组织，从数据到提交可完整复现：

| 阶段 | 脚本 | 主要产物 | 算力 |
|------|------|----------|------|
| **① 数据准备** | `00_load_and_clean.py` | `data/processed/wt.fasta`、`outputs/00_summary.json` | CPU |
| **② 位点分析** | `01_position_pool.py` | `position_pool.csv`、`lethal_blacklist.csv` | CPU |
| **③ 序列设计** | `02_seed_designs.py` | `seeds.csv`、`seeds.fasta` | CPU |
| **④ 合规与多样性** | `07_dnachisel_check.py`、`03_diversity_check.py` | `07_seed_check.csv`、`03_diversity_report.json` | CPU |
| **⑤ 机器学习** | `04_embed_esm.py` → `05_train_regressor.py` → `05b_predict_seeds.py` → `06b_generate_candidates.py` | 嵌入 `.npz`、模型 `.pkl`、候选 CSV | **GPU** |
| **⑥ 热稳定评估** | `08_thermompnn.py`、`08c_thermompnn_top200.py` | ThermoMPNN ΔΔG、替换建议 | **GPU** |
| **⑦ 提交生成** | `06_make_submission.py` | `submission.csv`（CRLF） | CPU |

设计思路详见 [docs/design_doc.md](./docs/design_doc.md)。

---

## 最终提交的 6 条序列

| Seq | 角色 | 母本 | 突变 | ML 亮度 ratio | ThermoMPNN ΔΔG |
|-----|------|------|------|---------------|----------------|
| 1 | 保底 | avGFP | `S65T:S72A` | 0.88 | -0.11 |
| 2 | 中稳 | avGFP | `S65T:S72A:K79R:N105Y:I167V` | 0.91 | -0.48 |
| 3 | sfGFP 对照 | sfGFP | `S72A` | 0.70 | -0.10 |
| 4 | 中稳 | avGFP | `S65T:S72A:V93F:L178V:A206V` | 0.90 | -0.52 |
| 5 | **冲金主炮** | avGFP | `S65T:S72A:K79R:L178V` | **1.25** | **-0.35** |
| 6 | **彩票（最亮）** | avGFP | `S65T:S72A:N105Y:S147N:I171S:L178V` | **1.27** | +1.05 |

完整序列见 `outputs/seeds.csv` / `outputs/submission.csv`。

---

## 环境配置

### 推荐运行环境

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（已在 Ubuntu 22.04 测试） |
| Python | **3.10** |
| GPU（阶段⑤⑥） | NVIDIA GPU + CUDA 12.1（V100 32GB 即可） |

### 一键安装

```bash
git clone https://github.com/LynnChrisleetop/ProteinDesign.git
cd ProteinDesign
bash scripts/setup_env.sh
```

脚本会安装 PyTorch、fair-esm、LightGBM、DnaChisel，并将 ThermoMPNN 等 clone 到 `third_party/`。

### 手动安装（pip）

```bash
pip install -r requirements.txt

# PyTorch（按平台选择，示例：CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 赛事数据路径

赛方数据 **不进 Git**。将官方压缩包解压后，目录内需包含 `GFP_data.xlsx` 与 `Exclusion_List.csv`。

**方式 A（推荐）**：放在仓库同级目录

```
../2026Protein Design/
├── GFP_data.xlsx
├── Exclusion_List.csv
├── AAseqs of 5 GFP proteins_20260511.txt
└── submission_template.csv
```

**方式 B**：任意路径 + 环境变量

```bash
export GFP_DATA_DIR="/path/to/2026Protein Design"
```

验证：

```bash
python -c "from src.utils import DATA_DIR; print(DATA_DIR)"
```

---

## 模型与依赖

| 组件 | 用途 | 说明 |
|------|------|------|
| **ESM2-35M** (`fair-esm`) | 序列嵌入 | 默认 `esm2_t12_35M_UR50D`，480 维 |
| **LightGBM** | 亮度回归（log10） | 训练集 Test R² ≈ 0.71 |
| **ThermoMPNN** | 热稳定 ΔΔG | `third_party/ThermoMPNN`，需 GPU |
| **DnaChisel** | 反向翻译 / 合成可行性 | 提交前自检 |

大文件（`*.npz` 嵌入、`*.pkl` 模型权重）在 `.gitignore` 中，**需运行阶段⑤重新生成**。

---

## 如何运行（复现指南）

以下命令均在仓库根目录执行：`cd ProteinDesign`

### 阶段 ①–④：数据 → 设计 → 合规（CPU）

```bash
python src/00_load_and_clean.py
python src/01_position_pool.py
python src/02_seed_designs.py
python src/07_dnachisel_check.py
python src/03_diversity_check.py
```

### 阶段 ⑤：机器学习 — 嵌入、训练、推理、候选（GPU）

```bash
python src/04_embed_esm.py --model t12_35M
python src/05_train_regressor.py
python src/05b_predict_seeds.py
python src/06b_generate_candidates.py --n-samples 5000 --top-k 20
```

**推理入口**（已有模型权重时）：

```bash
python src/05b_predict_seeds.py --model-pkl outputs/05_model_esm35m_lgbm.pkl
python src/06b_generate_candidates.py --model-pkl outputs/05_model_esm35m_lgbm.pkl
```

### 阶段 ⑥：热稳定评估（GPU）

```bash
python src/08_thermompnn.py
python src/08c_thermompnn_top200.py
```

### 阶段 ⑦：生成提交文件

```bash
python src/06_make_submission.py --team 新次元小队 --strict-check
```

输出：`outputs/submission.csv`（UTF-8 无 BOM，CRLF 行尾，表头 `Team_Name,Seq_ID,Sequence`）。

---

## 仓库结构

```
ProteinDesign/
├── README.md                 # 环境 + 复现 + 推理说明（本文件）
├── requirements.txt          # Python 依赖清单
├── 参赛指南.md                # 战略与规则解读
├── scripts/setup_env.sh      # 一键环境初始化
├── src/                      # 流水线脚本
├── data/processed/           # WT 等清洗产物
├── inputs/pdb/               # PDB（ThermoMPNN 用）
├── third_party/              # ThermoMPNN 等（setup 脚本 clone）
├── docs/design_doc.md        # 设计思路（可导出 PDF 提交）
└── outputs/                  # 实验产物与 submission.csv
```

---

## 复现性说明

- CPU / GPU 流水线在 Python 3.10 + Ubuntu 22.04 下测试通过。
- 赛方原始数据未在仓库中分发，请从赛方渠道获取。
- 随机种子：`06b_generate_candidates.py` 默认 `--rng-seed 42`；LightGBM `random_state=42`。

---

## 文档索引

| 文档 | 内容 |
|------|------|
| [README.md](./README.md) | 环境、依赖、运行方式（竞赛复现入口） |
| [参赛指南.md](./参赛指南.md) | 规则、策略、位点池方法论 |
| [docs/design_doc.md](./docs/design_doc.md) | 完整设计决策与数据分析 |

---

## License

Code: MIT · Documentation: CC BY-SA 4.0

**新次元小队** · LynnChrisleetop · 2026 合成生物学创新赛
