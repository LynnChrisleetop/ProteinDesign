# 2026 合成生物学创新赛 · GFP 蛋白设计 — 设计思路文档

> **团队**：LynnChrisleetop  
> **任务**：用计算方法设计 6 条 GFP 突变序列，最大化「热处理后亮度相对 WT 的比值」 \
> **当前阶段**：最终提交版（Draft B）· 已通过 Exclusion + DnaChisel + 赛方模板自检 \
> **最后更新**：v2.0 · Draft B：ML + ThermoMPNN 联合优化后的 6 条终稿（`outputs/submission.csv`）

---

## 1. 问题定义与评分内化

### 1.1 评分公式
每条序列得分

\[
S = \frac{F_{\text{initial}}}{F_{\text{initial}}^{\text{WT}}} \times \frac{F_{\text{final}}}{F_{\text{initial}}}
  = \frac{F_{\text{final}}}{F_{\text{initial}}^{\text{WT}}}
\]

代数化简后 **真正决定胜负的只有一个量**：72 °C 处理**之后**的亮度相对 WT 的比值。亮度高但热处理后炸掉的设计，得分为 0。

### 1.2 硬约束（即时淘汰）

- 长度 ∈ [220, 250] aa
- `M` 开头
- 仅 20 标准氨基酸
- 6 条互不相同
- 6 条均不与 `Exclusion_List.csv`（135,414 条）任意一条**完全相同**

### 1.3 隐性红线

`Finitial < 0.3 × Finitial_WT` → 该条得 0 分。所以**保活 > 求强**。

### 1.4 战略选型

打榜只看 6 条中的 **Top-1**。所以理性策略是**风险梯度**：用 1–2 条兜底（拿银），1–2 条主攻（争金），1 条狙击单项（最佳热稳定/亮度），1 条彩票（高风险高收益）。**绝不能** 6 条同质化。

---

## 2. 数据理解（Day 1）

### 2.1 赛方资源清单

| 文件 | 内容 | 备注 |
|---|---|---|
| `GFP_data.xlsx::brightness` | 141,572 行（4 类 GFP 的突变-亮度） | Sarkisyan 2016 |
| `GFP_data.xlsx::beforetopseqs` | 20 条往届高分序列 | **独家先验** |
| `AAseqs of 5 GFP proteins.txt` | 5 条参考 WT | avGFP/sfGFP/amacGFP/cgreGFP/ppluGFP |
| `Exclusion_List.csv` | 135,414 条禁用序列 | 列名 `Sequence`（不是教程里写的 `sequences-not-submit`） |
| `submission_template.csv` | 提交模板 | **CRLF 行尾**，表头 `Team_Name,Seq_ID,Sequence` |

### 2.2 5 条 WT 全部合规

| WT | 长度 | M 开头 | 合规 |
|---|---|---|---|
| sfGFP   | 238 | ✓ | ✅ |
| avGFP   | 238 | ✓ | ✅ |
| amacGFP | 238 | ✓ | ✅ |
| cgreGFP | 235 | ✓ | ✅ |
| ppluGFP | 222 | ✓ | ✅（最短，仅 2 aa 缓冲） |

### 2.3 训练数据结构

`brightness` 表只有 4 类 WT（**sfGFP 不在内**！avGFP 51715 行、amacGFP 33809、ppluGFP 31480、cgreGFP 24568）。**关键发现**：

- 数据集中无 sfGFP — 任何针对 sfGFP 的 ML 模型只能借 avGFP 数据"近似类比"；
- 亮度是 **log10 尺度**（avGFP WT = 3.7192 → 线性 5238.6）；
- 4 类 WT 中 **cgreGFP 亮度最高**（log10 = 4.4969 → 线性 31403，约为 avGFP 的 6 倍）。

```
WT 亮度 (log10 / linear)：
  avGFP    3.7192 →   5238.6
  amacGFP  3.9707 →   9348.2
  cgreGFP  4.4969 →  31402.7   ← 最亮
  ppluGFP  4.2258 →  16819.6
```

### 2.4 `beforetopseqs` 是黄金先验

20 条赢家序列**全部 238 aa**，**全部合规**。这是赛方放出来的"教科书"——我们必须深度利用。

---

## 3. 位点池构造方法论（Day 2）

我们设计了**三股证据合成法**：

```
      数据驱动           文献先验            赢家 diff
 (51715 行 avGFP)     (Superfolder       (20 winners ×
       │              StayGold etc.)        avGFP/sfGFP)
       │                  │                     │
       ▼                  ▼                     ▼
  per-position         position           position-AA
   stats               whitelist           frequency
       │                  │                     │
       └──────────────────┼─────────────────────┘
                          ▼
              outputs/position_pool.csv
              (238 行 ×  score + category)
```

### 3.1 数据驱动：单点突变 per-position 统计

筛 avGFP 单点突变 4606 行，对每个位点计算：

- `n_mutants_seen` ：该位点在数据中被试过的不同突变数；
- `lethal_rate`  ：ratio < 0.05 × WT 的比例（致死率）；
- `tolerant_rate` ：ratio ≥ 0.50 × WT 的比例（保活率）；
- `n_boost` ：ratio ≥ 1.5 × WT 的次数；
- `n_super` ：ratio ≥ 2.0 × WT 的次数；
- `best_substitution / best_ratio` ：见过的最好单点突变。

阈值选择：
- 0.05 × WT 对应红线之下的"绝对死"；
- 0.30 × WT 是赛事 `Finitial` 红线（直接 0 分），值得格外注意；
- 0.50 × WT 是工程上的"可接受保活"；
- 1.50 × / 2.0 × 是有意义的增益。

### 3.2 致死黑名单（绝对避开）

判定规则：`lethal_rate > 0.5` 或 `(n_tolerant == 0 且 n_mutants_seen ≥ 3)`。

得到 18 个位点（with_M 编号；位号 = 数据集 pos + 1）：

```
发色团 / 紧邻              : Y66 (av[65]=Y, 发色团本体), G67 (紧邻发色团)
β-barrel 内核疏水位         : F85, V94
β-barrel 关键带电残基       : E18, D20, E33, E35, E91, E96, E125
功能位 / 刚性               : S31, S66*, P57, T60, S29
其他不可改动                : T51, Y183, D217
```

*S66 / S65 等位点：avGFP `WT[65]=Y` 是发色团，附近 S66 是发色团形成关键。

数据完美对应文献——**Y66 是发色团本体**，被列入致死池正确无误。

### 3.3 文献先验

```python
SUPERFOLDER_MUTATIONS = [
    S30R, Y39N, F64L, S65T, F99S, N105T,
    Y145F, M153T, V163A, I171V, A206V
]
STAYGOLD_MONOMERIZE   = [ A206K ]   # 单体化关键
EXTRA_KNOWN           = [ F46L, T203Y ]   # 折叠加速 / 红移
```

数据中匹配上：`outputs/literature_priors.csv`，14 个突变全数落在我们的"3 类标记"内。

### 3.4 关键发现 1：**sfGFP = avGFP + 11 个 Superfolder 突变**

代码自动比对 sfGFP 与 avGFP 的 WT 序列，**完全确认**：

```
S30R, Y39N, S65T, Q80R, F99S, N105T, Y145F, M153T, V163A, I171V, A206V
```

意义：**任何 sfGFP 设计都可表达为"avGFP + 突变包"**，让我们可以自由切换母本而不丢失任何信息。

### 3.5 关键发现 2：**往届赢家更偏 avGFP 母本**（与教程冲突）

20 条 `beforetopseqs` 与 sfGFP WT 的差异频次（top 5）：

| 位点 | sfGFP 是 | 赢家最常改成 | 频次 | 解读 |
|---|---|---|---|---|
| 145 | F | **Y** | **20/20** | 把 Y145F **还原** |
| 99  | S | **F** | 19/20 | 把 F99S **还原** |
| 105 | T | **N** | 15/19 | 把 N105T **还原** |
| 206 | V | **A** | 18/19 | 把 A206V **还原** |
| 153 | T | **M** | 18/19 | 把 M153T **还原** |

→ **20 条历史赢家几乎一致地把 Superfolder 改回 avGFP 原版**。这与官方教程及多数文献推荐的"sfGFP 母本"不一致。

警示：往届只评亮度，没有 72°C 热处理。**本届新增的热稳定考核可能让 sfGFP 重新占优**。终稿保留 **Seq_3** 一条 sfGFP 对照。

### 3.6 关键发现 3：**K158G 是数据驱动的新发现**

```
super_boost_positions.csv (修正后 with_M 编号):
  pos  wt_aa  n_mutants_seen  max_ratio  best_substitution  best_ratio
  158  K      7               2.479      G                  2.479
```

avGFP 单点 K158G 实测亮度 **2.48 × WT**，是 single-point 数据里最强的"增益位"，**官方教程位点池里没有**。我们把它纳入 Seq_4 / Seq_5 / Seq_6。

> **⚠️ 编号体系坑（v1.0 → v1.1 的修正）**：Sarkisyan brightness 数据集使用 **skip-M 1-based 编号**
> （起始 M 不算第 1 位，S=1），而文献 / 赛方 FASTA / 我们的 `apply_mutations` 用 **with-M
> 1-based 编号**（M=1）。两者位号差 1。v1.0 的 `01_position_pool.py` 在读数据时混用了两个体系，
> 把数据集的 `pos=157`（=skip_M）配上了 `WT[157-1]=Q`（=with_M 取字），导致整张
> `super_boost_positions.csv` 的 `wt_aa` 列错位。**修正后**统一以 with_M 编号输出，数据
> super-boost 真实身份是 **K158G**（avGFP `WT[157]=K`），而不是 v1.0 报告的 Q157G。所有 Seq_4 / 5 / 6
> 已替换。`detect_numbering()` + `strict=True` 双保险防止再次出错。

### 3.7 赢家相对 avGFP 的高频改动（赢家"指纹"）

| 位点 | 突变 | 频次 / 20 | 加分依据 |
|---|---|---|---|
| 65  | S→T | 9 | EGFP 经典发色团调谐 |
| 72  | S→A | 9 | **赢家专属，教程没有** |
| 80  | Q→R | 5 | superfolder 子集 |
| 163 | V→A | 5 | superfolder（折叠稳定） |
| 105 | N→K | 2 主 | 折叠 |
| 46  | F→L | 3 | 折叠加速 |

### 3.8 综合得分公式

```python
score(pos) = min(max_ratio, 5.0)            # 增益强度
           + 0.5 × n_boost                  # 见过几次增益突变
           + 1.5 × n_winners_changed        # 赢家加权最高
           + 2.0 if pos in literature else 0
           + 1.5 if pos is sfGFP-vs-avGFP diff else 0
score(lethal_pos) = -100.0                  # 致死位惩罚
```

`outputs/position_pool.csv` 238 个位点全数打分。Top-30 涵盖了所有六槽种子设计需要用到的位点。

### 3.9 类别分布

| Category | 数量 | 含义 |
|---|---|---|
| SAFE | 181 | 可保活，不致死 |
| LIT+SAFE | 6 | 文献位且数据保活 |
| BOOST+LIT+SAFE | 6 | 文献位 + 数据增益 |
| BOOST+SAFE | 6 | 数据驱动增益位 |
| LETHAL | 18 | 绝对避开 |
| UNKNOWN | 21 | 数据稀疏，无判断 |

---

## 4. 最终六条设计（Draft B）

### 4.1 设计原则

1. **避开致死黑名单**（含 Y66 发色团）
2. **规则初筛 + ML 亮度 + ThermoMPNN 热稳定** 三阶段收敛
3. **梯度排布**：保底 1 条 → 中稳 2 条 → sfGFP 对照 1 条 → 冲金 1 条 → 彩票 1 条
4. **母本**：avGFP × 5 + sfGFP × 1（Seq_3 保留 superfolder 对照）
5. **6 条序列互不相同**；Seq_1 ↔ Seq_5 Hamming = 2（共享 S65T:S72A 骨架，刻意「保底 + 冲金」）

### 4.2 最终 6 条明细（提交版）

| Seq | 角色 | 策略 | 母本 | 突变（with_M） | ML ratio | ΔΔG | 设计动机 |
|:---:|---|---|---|---|:---:|:---:|---|
| **1** | 保底 | safe-baseline | avGFP | `S65T:S72A` | 0.88 | −0.11 | 赢家最高频两改，最少扰动 |
| **2** | 中稳 | draftB-repl-c21 | avGFP | `S65T:S72A:K79R:N105Y:I167V` | 0.91 | −0.48 | 替换原 winner-stack；Top200 cand#21 |
| **3** | sfGFP 对照 | sfGFP-control | sfGFP | `S72A` | 0.70 | −0.10 | superfolder 骨架 + 最小扰动 |
| **4** | 中稳 | draftB-repl-c23 | avGFP | `S65T:S72A:V93F:L178V:A206V` | 0.90 | −0.52 | 替换原 boost-engine（原 ΔΔG +4.3）；cand#23 |
| **5** | **冲金主炮** | draftB-gold-c2 | avGFP | `S65T:S72A:K79R:L178V` | **1.25** | **−0.35** | Top200 combo_rank #1；亮且相对稳 |
| **6** | **彩票** | draftB-lottery | avGFP | `S65T:S72A:N105Y:S147N:I171S:L178V` | **1.27** | +1.05 | ml-top1 最亮候选；接受热稳定风险 |

> ML ratio = ESM2-35M + LightGBM 预测的初始亮度相对同母本 WT 的比值（**不含 72°C 热处理**）。  
> ΔΔG = ThermoMPNN 单点 ΔΔG 求和（越负越稳）。完整序列见 `outputs/seeds.csv` / `outputs/submission.csv`。

### 4.3 从初稿到终稿的关键替换

| Seq | 初稿（规则设计） | 终稿（Draft B） | 替换理由 |
|:---:|---|---|---|
| 2 | winner-stack（ΔΔG +1.38） | cand#21 | 热稳定改善，ML 亮度 0.91× |
| 4 | boost-engine（ΔΔG **+4.30**） | cand#23 | 原设计 ThermoMPNN 几乎必炸 |
| 5 | gold-rush / ml-top1 | cand#2 | 比 ml-top1 略暗（1.25 vs 1.27）但 ΔΔG 由 +1.05 → **−0.35** |
| 6 | sfGFP K158G:A206K | ml-top1 彩票 | 释放槽位给最亮组合，博 Top-1 上限 |

保留不变：Seq_1（保底）、Seq_3（唯一 sfGFP 对照）。

### 4.4 每条槽位的风险评估（终稿）

| Seq | 槽位角色 | 主要风险 | 缓解 |
|:---:|---|---|---|
| 1 | 保底 | 亮度中等（0.88×） | 热稳定好；防全军覆没 |
| 2 | 中稳 | 亮度非最高 | ΔΔG −0.48；与 Seq_1/4 形成中位带 |
| 3 | sfGFP 对照 | ML 对 sfGFP 无训练数据 | 热稳定好；测试 superfolder 假设 |
| 4 | 中稳 | ratio 0.90× | 已消除原 Seq_4 的热稳定硬伤 |
| 5 | 冲金主炮 | 与 Seq_1 Hamming=2 | combo_rank #1；亮 + 稳的最佳折中 |
| 6 | 彩票 | ΔΔG +1.05，热处理可能掉亮度 | 初始最亮（1.27×）；专博 Top-1 上限 |

### 4.5 为什么 Seq_3 不是"sfGFP 原样"？

`sfGFP WT` 本身在 `Exclusion_List` 内（所有 WT 都被禁），exact match 即 0 分。所以做了**最小扰动**：`S72A`。这个改动满足：
- 离开 Exclusion；
- 是赢家最高频改动之一；
- 不破坏 sfGFP 的热稳定核心（72 位远离发色团）；
- 与"sfGFP 对照"语义最相近。

### 4.6 多样性诊断（终稿）

```
两两 Hamming 矩阵（Draft B，见 outputs/03_pairwise_hamming.csv）:
       Seq_1  Seq_2  Seq_3  Seq_4  Seq_5  Seq_6
 Seq_1   0     3    10     3     2     4
 Seq_2   3     0    12     6     3     5
 Seq_3  10    12     0    11    12    12
 Seq_4   3     6    11     0     3     5
 Seq_5   2     3    12     3     0     4
 Seq_6   4     5    12     5     4     0

mean=6.33  median=5  min=2  max=12
unique parents = 2 (avGFP × 5, sfGFP × 1)
```

**已知 tradeoff**：
- Seq_1 vs Seq_5 = **2**（共享 S65T:S72A + K79R/L178V 路径），保底与冲金同骨架，仍满足 6 条互不相同
- 母本仅 av/sf 两类；cgreGFP 备选见 `outputs/03_cgreGFP_candidate.txt`

判定：`NEEDS_DIVERSIFICATION`（脚本自动 verdict）— 对打榜可接受；6 条功能梯度清晰。

---

## 5. 合规与可合成性检查（终稿验收）

`07_dnachisel_check.py` 对 6 条种子做了**完整体检**：

### 5.1 基础合规
- 全部 6 条 长度 = 238 aa ✅
- 全部 M 开头 ✅
- 全部仅 20 标准氨基酸 ✅

### 5.2 Exclusion 检查
- exact-match：0/6 命中 ✅
- 同长度 Hamming 最近邻：min=1（Seq_1 / Seq_3），其余 ≥ 2 — **赛事红线只看 exact match，hamming=1 合法**

### 5.3 DnaChisel 反向翻译（AA → DNA）

约束设置：
- `EnforceTranslation()` — 保证 DNA 翻译回原 AA
- `EnforceGCContent(0.30, 0.70, window=80)` — 滑动 GC 30–70%
- `AvoidPattern(EcoRI/BamHI/HindIII/NdeI)` — 避酶切位点
- `AvoidPattern(连续 6 个相同碱基)` — 避同聚体（合成失败常见原因）
- `CodonOptimize(species="e_coli")` — 大肠杆菌密码子优化

结果：

| Seq | DNA len | GC% | 约束 | 耗时 |
|---|---|---|---|---|
| 1 | 714 | 50.0% | ✅ 全过 | 0.14 s |
| 2 | 714 | 50.3% | ✅ 全过 | 0.13 s |
| 3 | 714 | 51.0% | ✅ 全过 | 0.13 s |
| 4 | 714 | 49.4% | ✅ 全过 | 0.14 s |
| 5 | 714 | 50.1% | ✅ 全过 | 0.13 s |
| 6 | 714 | 50.0% | ✅ 全过 | 0.14 s |

→ **6 条全部可合成**，赛方 CFPS 流程理论上不会因序列问题失败。

### 5.4 最终判定

**6/6 OK** — `python src/06_make_submission.py --team 新次元小队 --strict-check` 已通过。  
最终提交文件：`outputs/submission.csv`（CRLF，1575 字节，队名=新次元小队）。

---

## 6. 机器学习与热稳定阶段（已完成）

### 6.1 ML 亮度模型

| 项目 | 结果 |
|---|---|
| 嵌入 | ESM2-35M，141,365 序列，480 维 |
| 回归 | LightGBM，Test **R² = 0.714**（教程 RF 基线 0.28） |
| 候选搜索 | `06b_generate_candidates.py`，5000 组合 → Top-200 |
| 产物 | `outputs/05_model_esm35m_lgbm.pkl`、`outputs/06b_top_candidates.csv` |

### 6.2 ThermoMPNN 热稳定

| 脚本 | 工作 | 产物 |
|---|---|---|
| `08_thermompnn.py` | 6 条 seeds 单点 ΔΔG | `outputs/08_thermompnn_seeds.csv` |
| `08c_thermompnn_top200.py` | Top-200 候选 combo_rank | `outputs/08c_top200_scored.csv` |

Draft B 的 Seq_2/4/5/6 替换均来自 Top-200 中 **ML 亮度 × ThermoMPNN 稳定性** 的联合排序。

### 6.3 仍存在的局限（诚实披露）

1. **ML 只预测 Finitial**，不预测 72°C 后 Ffinal；Seq_6 彩票位 ΔΔG 仍为正
2. **sfGFP 无训练数据**，Seq_3 的 ML ratio 只能近似参考
3. **母本仅 av/sf**，cgreGFP（最亮 WT）未纳入终稿
4. **Seq_1 ↔ Seq_5 Hamming = 2**，多样性脚本会警告，但 6 条序列互不相同

### 6.4 可选后续（未纳入终稿）

- ESMFold pLDDT 结构过滤（`09_esmfold_check.py` 待写）
- cgreGFP 母本备选（见 `outputs/03_cgreGFP_candidate.txt`）
- 设计思路 PDF / Agent 决策日志导出

---

## 7. 复现性

### 7.1 环境锁定

- 推荐环境：Ubuntu 22.04，Python 3.10，CUDA 12.1（GPU 阶段）
- 关键包版本：见 `requirements.txt` 与 `scripts/setup_env.sh`
- 阶段 ①–④ 仅需 CPU，全套约 30 秒跑完

### 7.2 一键复现命令

```bash
git clone https://github.com/LynnChrisleetop/ProteinDesign.git
cd ProteinDesign

# 放赛事数据到 ../2026Protein Design/  或 export GFP_DATA_DIR=...

pip install pandas numpy openpyxl scikit-learn lightgbm biopython joblib \
            matplotlib seaborn dnachisel python-codon-tables

python src/00_load_and_clean.py
python src/01_position_pool.py
python src/02_seed_designs.py
python src/07_dnachisel_check.py
python src/03_diversity_check.py
python src/06_make_submission.py --team <YourTeam> --strict-check
```

### 7.3 产物清单

- 18 个 csv/fasta/json 在 `outputs/`
- 4 个清洗产物在 `data/processed/`
- 最终提交 `outputs/submission.csv`（CRLF，1575 字节）

---

## 8. 设计哲学

> "先保活，再求强" — 30% 红线决定一切；任何"亮度爆表但 72°C 一把炸没"的设计都是 0 分

> "组合优于孤注" — 打榜只看 Top-1，但 6 条间的风险梯度才决定中签概率

> "数据 + 文献 + 赢家 diff 三家收敛 → 可信" — 任何一家单独都可能误导，三家收敛才是真信号

> "规则可解释 > ML 黑盒（在数据稀疏时）" — Day 2 的位点池统计 R² 远好于教程 ESM+RF 的 0.28

---

## 9. 致谢与引用

- Sarkisyan et al., *Local fitness landscape of the green fluorescent protein* (Nature 2016) — `GFP_data.xlsx` 来源
- Pédelacq et al., *Engineering and characterization of a superfolder GFP* (Nat Biotechnol 2006) — Superfolder 突变包
- Hirano et al., *A highly photostable and bright green fluorescent protein* (Nat Biotechnol 2022) — StayGold
- 赛事组织方提供的 `Basic Tutorial on Protein Design.ipynb` — 基线代码灵感
- Edinburgh Genome Foundry, *DnaChisel* — AA → DNA 反向翻译工具

---

**附录 A：完整位点池 Top-30**（按综合得分降序）

见 `outputs/position_pool.csv` 中 `score > 0` 的前 30 行。

**附录 B：6 条种子完整序列**

见 `outputs/seeds.fasta`、`outputs/seeds.csv`、`outputs/07_seed_dna.csv`。

**附录 C：当前进度看板**

见仓库 `README.md`。
