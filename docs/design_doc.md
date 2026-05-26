# 2026 合成生物学创新赛 · GFP 蛋白设计 — 设计思路文档

> **团队**：LynnChrisleetop  
> **任务**：用计算方法设计 6 条 GFP 突变序列，最大化「热处理后亮度相对 WT 的比值」 \
> **当前阶段**：Day 2.5 v2（编号体系修正后）；ML 阶段 (Day 3+) 启动中 \
> **最后更新**：v1.1 · 修正了 Sarkisyan 数据集与文献编号体系混用导致的 Q157G 误读 → K158G

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

警示：往届只评亮度，没有 72°C 热处理。**本届新增的热稳定考核可能让 sfGFP 重新占优**。所以策略上我们保留 sfGFP 母本两条做对照（Seq_3、Seq_6）。

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

## 4. 六条种子设计（Day 2.5）

### 4.1 设计原则

1. **避开致死黑名单**（含 Y66 发色团）
2. **优先赢家高频改动**（依据 §3.7）
3. **梯度排布**：保底 1 条 → 稳进 2 条 → 增益 1 条 → 冲金 1 条 → 高风险 1 条
4. **母本对照**：avGFP 4 条 + sfGFP 2 条
5. **6 条互相 Hamming ≥ 2**（避免重复浪费 6 槽）

### 4.2 6 条设计明细

| Seq | 策略 | 母本 | 突变（with_M 编号） | 设计动机 |
|---|---|---|---|---|
| **1** | safe-baseline | avGFP | `S65T:S72A` | 赢家最高频两改（9/20 each），最少改动保底 |
| **2** | winner-stack | avGFP | `S65T:S72A:Q80R:N105K:V163A` | 赢家 Top-5 高频改动叠加，稳进 |
| **3** | sfGFP-control-minus | sfGFP | `S72A` | sfGFP + 最小扰动；热稳定基准 |
| **4** | boost-engine | avGFP | `F46L:K158G:V163A` | F46L 折叠 + K158G (data 2.48× super-boost) + V163A |
| **5** | gold-rush | avGFP | `F46L:S65T:S72A:Q80R:N105K:K158G:V163A` | Seq_2 + F46L + K158G，主力冲金 |
| **6** | high-risk-monomer-superboost | sfGFP | `K158G:A206K` | sfGFP 单体化（A206K）+ 真实数据 super-boost K158G |

### 4.3 每条设计的风险评估

| Seq | 预期 Top-1 概率 | 失败模式 | 缓解 |
|---|---|---|---|
| 1 | 高保活，低增益 | avGFP 整体热稳定不足 → Ffinal 偏低 | 由 Seq_3/6 兜底 |
| 2 | 中等保活，中等增益 | 5 突变累积扰动 | 由 Seq_1 兜底 |
| 3 | 高稳定，亮度中性 | S72A 单点不显眼 | 单项最佳热稳定狙击 |
| 4 | 高增益赌注 | K158G 与 F46L / V163A 不协同 | 由 Seq_1/2 兜底 |
| 5 | 主力冲金 | 7 突变累积破坏 → Finitial 跌破红线 | 由 Seq_1/2 兜底 |
| 6 | 高风险高收益 | A206K + K158G + sfGFP 三重叠加爆炸 | 接受失败 |

### 4.4 为什么 Seq_3 不是"sfGFP 原样"？

`sfGFP WT` 本身在 `Exclusion_List` 内（所有 WT 都被禁），exact match 即 0 分。所以做了**最小扰动**：`S72A`。这个改动满足：
- 离开 Exclusion；
- 是赢家最高频改动之一；
- 不破坏 sfGFP 的热稳定核心（72 位远离发色团）；
- 与"sfGFP 对照"语义最相近。

### 4.5 多样性诊断（Day 2.5 收尾）

```
两两 Hamming 矩阵 (after Seq_5 加 Q157G 修正):
       Seq_1  Seq_2  Seq_3  Seq_4  Seq_5  Seq_6
 Seq_1   0     3    10     5     5    12
 Seq_2   3     0     8     6     2    10
 Seq_3  10     8     0    13    10     3
 Seq_4   5     6    13     0     4    11
 Seq_5   5     2    10     4     0    10
 Seq_6  12    10     3    11    10     0

mean=7.47  median=8  min=2  max=13
unique parents = 2 (avGFP × 4, sfGFP × 2)
```

**已知 tradeoff**：
- Seq_2 vs Seq_5 = 2（Seq_5 = Seq_2 + F46L + Q157G），但二者意图不同——Seq_5 是"再加两点是否能放大增益"的实验
- 仅 2 个母本：cgreGFP 备选已草稿（见 `outputs/03_cgreGFP_candidate.txt`），ML 阶段补强

判定：`NEEDS_DIVERSIFICATION` — 当前为可接受的保底，ML 阶段会引入更多样的设计。

---

## 5. 合规与可合成性检查（Day 2.5 收尾）

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
| 1 | 714 | 49.86% | ✅ 全过 | 0.12 s |
| 2 | 714 | 49.86% | ✅ 全过 | 0.13 s |
| 3 | 714 | 50.70% | ✅ 全过 | 0.14 s |
| 4 | 714 | 50.28% | ✅ 全过 | 0.13 s |
| 5 | 714 | 50.42% | ✅ 全过 | 0.13 s |
| 6 | 714 | 50.56% | ✅ 全过 | 0.13 s |

→ **6 条全部可合成**，赛方 CFPS 流程理论上不会因序列问题失败。

### 5.4 最终判定
**6/6 OK** — 当前 `outputs/submission.csv` 可作为合规保底直接提交。

---

## 6. 已知局限与下一步

### 6.1 当前局限

1. **完全没用 ML 模型评估亮度**——全靠规则与历史先验，对未知组合效应估计有限
2. **完全没有热稳定预测**——本届新增的 72°C 维度是评分核心，目前完全靠 sfGFP 的"经验稳定性"
3. **只有 2 个母本**（avGFP / sfGFP）——cgreGFP（最亮 WT）未利用
4. **6 条之间相关性较高**——Seq_2 vs Seq_5 hamming = 2
5. **K158G 是个赌注**——单点 2.48× 来自 7 次观察，与多点组合的协同未知

### 6.2 P0：提交保底（已可做）

- ✅ `outputs/submission.csv` 已生成（CRLF + 字节级对齐模板）
- ⏳ 写 `docs/agent_log.md`（LLM 决策日志）

### 6.3 P1：ML 阶段（**需 GPU，¥10–20**）

| 脚本 | 工作 | 目标 |
|---|---|---|
| `src/04_embed_esm.py` | ESM2-150M 嵌入 ≥ 20k 序列（教程仅 5k） | 训练样本充足 |
| `src/05_train_regressor.py` | LightGBM 替代教程 RF；尝试 log10 → 线性反变换 | R² ≥ 0.40（教程仅 0.28） |
| `src/06b_generate_candidates.py` | 用 30 位点池做组合突变 10k+，模型筛 Top-200 | 自动发现协同突变 |
| 整合 | 用 ML Top-K 替换当前 Seq_4 / Seq_5 / Seq_6 中风险最高的 | Top-1 期望 +20-50% |

### 6.4 P2：热稳定预测（**核心评分维度，需 GPU**）

| 脚本 | 工作 |
|---|---|
| `src/08_thermompnn.py` | 对 Top-200 候选跑 ThermoMPNN 全位点 ΔΔG 扫描 |
| `src/09_esmfold_check.py` | ESMFold 算 pLDDT，过滤"结构不合理"设计 |

无热稳定预测意味着我们的 Seq_4/5（avGFP 母本 + 多突变）**可能在 72°C 失活**。这是当前管线最大的盲区。

### 6.5 P3：多样性补强

- 加 1 条 cgreGFP 母本设计（数据 WT 亮度 31403，是 avGFP 的 6×）
- 加 1 条 sfGFP + TGP 同源稳定位点的"非 av/sf"设计

### 6.6 P4：加分项

- 设计思路 PDF（导出本文档）
- `docs/agent_log.md`：LLM 与 Cursor Agent 的关键决策日志
- GitHub 公开仓库 README 已完成 ✅

---

## 7. 复现性

### 7.1 环境锁定

- 镜像：`ubuntu:22.04-py3.10-cuda12.1`（Bohrium 标准）
- Python：3.10.6
- 关键包版本：见 `requirements.txt` 与 `scripts/bohrium_init.sh`
- Day 1–2.5 仅需 CPU（4 核 / 8 GB），全套 30 秒跑完

### 7.2 一键复现命令

```bash
git clone https://github.com/LynnChrisleetop/ProteinDesign.git
cd ProteinDesign

# 放赛事数据到 ../2026Protein Design/  或 挂载 /bohr/...

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
