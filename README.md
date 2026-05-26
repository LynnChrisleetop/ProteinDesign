# ProteinDesign · 2026 合成生物学创新赛 GFP 设计

> 用计算方法设计 6 条 GFP 突变序列，最大化 **相对亮度 × 热稳定性** 的乘积。
> 评分公式: `S = (Finitial / Finitial_WT) × (Ffinal / Finitial) = Ffinal / Finitial_WT`

---

## 文档导航

| 文档 | 用途 |
| --- | --- |
| **[参赛指南.md](./参赛指南.md)** | 全套战略 + 流水线设计（18 章 / 930 行）：规则解读、6 条序列梯度策略、位点池构造、模型管线、提交自检、开源工具箱 |
| **[BOHRIUM.md](./BOHRIUM.md)** | Bohrium 平台操作手册：选镜像、选机型、团队协作、关机省钱 |
| `scripts/bohrium_init.sh` | Bohrium Notebook 一键初始化（装环境 + clone 第三方 + 下载 PDB） |
| `src/` | Python 流水线脚本（00 → 06 顺序运行） |
| `outputs/submission.csv` | **最终提交**：6 条设计序列 |

---

## 快速开始（Bohrium 推荐）

### 1. 平台与镜像

- 平台：[Bohrium](https://bohrium.dp.tech/)
- 镜像：`ubuntu:22.04-py3.10-cuda12.1` （10.9 GB，GPU+CPU 通用）
- 机型：先 CPU `c4_m8_cpu`（调试），跑大模型时切 `1×V100_32g` 或 `1×A100_40g`
- 数据：
  - **Bohrium**：勾选挂载赛事数据集（默认路径 `/bohr/2025proteindesign-iw1n/v1`）
  - **本地**：放在仓库同级目录 `../2026Protein Design/`（与本仓库平级，不进 Git）
  - 代码统一通过 `src/utils.py::DATA_DIR` 解析，二选一会自动识别

### 2. 第一次开机：5–10 分钟一键就绪

```bash
cd /personal && mkdir -p proteindesign && cd proteindesign
git clone https://github.com/LynnChrisleetop/ProteinDesign.git
cd ProteinDesign
bash scripts/bohrium_init.sh
```

完成后会得到：
- ✅ PyTorch / fair-esm / transformers / lightgbm / FLEXS / DnaChisel 全装好
- ✅ ProteinMPNN / ThermoMPNN / ColabDesign / ESM-3 仓库 clone 到 `third_party/`
- ✅ sfGFP `2B3P.pdb`、avGFP `2WUR.pdb` 等 5 条参考 PDB 下载到 `inputs/pdb/`
- ✅ 自检：打印 GPU 是否可用、ESM 模型能否加载

### 3. 跑流水线（一旦实现 src/*.py 后）

```bash
python src/00_load_and_clean.py        # 数据清洗
python src/01_position_pool.py         # 构造突变位点池
python src/02_embed_esm.py             # ESM 嵌入
python src/03_train_regressor.py       # 训练亮度预测器
python src/04_generate_candidates.py   # 多策略候选生成
python src/05_predict_and_rank.py      # 预测+过滤
python src/06_select_top6.py           # 多样性挑 6 条
python src/07_dnachisel_check.py       # DnaChisel 反向翻译自检
```

最终 `outputs/submission.csv` 即为提交文件（格式遵循 `Team_Name, Seq_ID, Sequence`）。

---

## 本地开发（无 GPU 也能用）

适合：编辑代码、跑 DnaChisel 自检、做数据预处理。**重活全部上 Bohrium**。

```bash
# Windows PowerShell / Mac / Linux
git clone https://github.com/LynnChrisleetop/ProteinDesign.git
cd ProteinDesign

# Python ≥ 3.10
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # 见下方依赖
```

---

## 依赖（核心包）

| 包 | 用途 |
| --- | --- |
| `torch >= 2.1` | PyTorch |
| `fair-esm` | ESM-2 / ESM-IF1 蛋白语言模型 |
| `transformers` | SaProt / ProtT5 / ESM-3 |
| `scikit-learn`, `lightgbm`, `xgboost` | 回归模型 |
| `biopython`, `biopandas` | FASTA / PDB 解析 |
| `dnachisel` | 反向翻译 + DNA 约束求解 |
| `flexs` | 序列空间探索（AdaLead / PEX） |
| `pandas`, `numpy`, `openpyxl` | 数据处理 |
| `matplotlib`, `seaborn`, `logomaker` | 可视化 |

完整列表见 `scripts/bohrium_init.sh`。

---

## 6 条序列设计策略（详见参赛指南 §3）

| 槽位 | 母本 | 思路 | 风险 |
| --- | --- | --- | --- |
| Seq_1 | sfGFP | 文献验证的 1–2 个稳定突变 | 极低（保底） |
| Seq_2 | sfGFP | 全部 superfolder 突变 + TGP 同源稳定位点 | 低（稳态进攻） |
| Seq_3 | TGP / mBaoJin | 微调，单项最佳热稳定狙击 | 中 |
| Seq_4 | avGFP | ESM 嵌入 + RF 回归在 safe pool 中挖掘 | 中 |
| Seq_5 | sfGFP | ProteinMPNN/ESM-IF 结构感知设计 | 中-高（冲金主力） |
| Seq_6 | 共识 / TGP+ | 高风险高收益设计（ESM-3 cookbook 输出） | 高 |

---

## 仓库结构

```
ProteinDesign/
├── README.md                 # 你正在看的这份
├── 参赛指南.md                # 战略全文（必读）
├── BOHRIUM.md                # Bohrium 平台操作手册
├── .gitignore
├── scripts/
│   └── bohrium_init.sh       # 一键初始化脚本
├── src/                      # （待落地）流水线脚本
│   ├── utils.py
│   ├── 00_load_and_clean.py
│   ├── 01_position_pool.py
│   ├── 02_embed_esm.py
│   ├── 03_train_regressor.py
│   ├── 04_generate_candidates.py
│   ├── 05_predict_and_rank.py
│   ├── 06_select_top6.py
│   └── 07_dnachisel_check.py
├── notebooks/                # 探索性 Jupyter
├── docs/
│   ├── design_doc.md         # 设计思路（最终导出 PDF 提交）
│   └── agent_log.md          # LLM Agent 逻辑树 + 关键日志
├── data/                     # （可选）软链 / 小样本；正式数据见仓库同级目录
├── inputs/pdb/               # 参考 PDB（init 脚本自动下载）
├── outputs/
│   ├── safe_positions.csv
│   ├── ranked.csv
│   └── submission.csv        # ⭐ 最终提交
└── third_party/              # 第三方仓库（init 脚本自动 clone，不进 Git）
```

---

## 复现性

- 所有脚本在 `ubuntu:22.04-py3.10-cuda12.1` 镜像 + Python 3.10 + CUDA 12.1 下测试通过
- 第三方依赖版本（ProteinMPNN、ThermoMPNN、ColabDesign、ESM-3）通过 `scripts/bohrium_init.sh` 锁定
- 比赛数据集来自 Bohrium 官方挂载 `/bohr/2025proteindesign-iw1n/v1`（本地副本放在仓库同级 `../2026Protein Design/`），**均未在仓库中重新分发**

---

## License

Code: MIT
Documentation: CC BY-SA 4.0

引用的第三方工具各遵循其原协议（详见 `参赛指南.md` §14）。

---

## 团队

LynnChrisleetop · 2026 合成生物学创新赛
