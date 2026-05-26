# Bohrium 工作手册（开实例 → 跑实验 → 停机一条龙）

> 这是 GFP Protein Design 比赛在 Bohrium 平台上的官方操作流程。**所有队员开新实例时都按本文操作**，避免环境不一致 / 钱花冤枉。

---

## 0. 一句话流程

```
登录 Bohrium → 选镜像 ubuntu:22.04-py3.10-cuda12.1 → 选机型（先 CPU 后 GPU）
  → 挂载比赛数据集 → 进 Notebook → 第一次跑 bash scripts/bohrium_init.sh
  → 之后每次只 cd /personal/proteindesign/ProteinDesign && git pull
  → 实验做完立刻 Stop 节点
```

---

## 1. 镜像（统一锁定一份，别换）

**全队统一用：** `ubuntu:22.04-py3.10-cuda12.1` （10.9 GB，GPU+CPU 都能跑）

| 为什么是它 | 说明 |
| --- | --- |
| Python 3.10 | 兼容 fair-esm / transformers / SaProt / ProteinMPNN / ThermoMPNN |
| CUDA 12.1 | 现代 PyTorch ≥ 2.1 都能用 |
| GPU+CPU 通用 | 切换机型时不用换镜像，环境保持一致 |
| 体积小（10.9 GB）| 启动快，不像 79 GB 的 bohrium-notebook 要等很久 |

> ❌ **不要选**：`bohrium-notebook:*`（含 DeePMD/TF 我们用不到）、`*-py3.7-*`（Python 3.7 装不了 ESM-3）、`*-py3.12-*`（fair-esm 不兼容）、`*-pytorch2.0`（CPU only）、`*-R*`（R 用不到）。

---

## 2. 机型（按任务阶段选，能小别大）

### 档位 A：`c4_m8_cpu`  — 默认调试机
- 4 核 CPU + 8 GB 内存，**< ¥0.3/h**
- **90 % 时间开这个**：写代码、数据预处理、DnaChisel、读 Excel、git push/pull、看 notebook
- ⚠️ 别一上来就开 GPU，**烧钱凶**

### 档位 B：`1×NVIDIA V100_32g`  — 主力训练机
- V100 32 GB 显存，**¥4–7/h**
- 用途：ESM-2 嵌入、RF/LightGBM 训练、ThermoMPNN 全位点扫描、ProteinMPNN 100 条重设计、SaProt 嵌入
- **70 % 的 GPU 时间用这个**

### 档位 C：`1×NVIDIA A100_40g` 或 `_80g`  — 决战机
- A100 40/80 GB 显存，**¥8–15/h**
- 用途：**ESM-3 7B / Boltz-1 / Chai-1 / 批量 ESMFold**
- 比赛末期 1–2 次冲刺，每次 1–2 小时即可

### 任务 → 机型 速查表

| 阶段 | 任务 | 机型 | 大约耗时 |
| --- | --- | --- | --- |
| Day 0–1 | 装环境 / 跑 DnaChisel / 数据清洗 | A | 不限 |
| Day 2 | 构造位点池（pandas + 文献统计） | A | 1 h |
| Day 3 | ESM-2 嵌入 (5 k 序列) + RF 训练 | **B** | 1.5 h |
| Day 3 | ThermoMPNN 全位点扫描（sfGFP） | **B** | 0.5 h |
| Day 4 | 候选生成 1 万条（纯逻辑） | A | 0.5 h |
| Day 4 | 候选嵌入 1 万条 + RF 预测 | **B** | 2 h |
| Day 4 | ProteinMPNN 100 条重设计 | **B** | 0.5 h |
| Day 5 | DnaChisel 反向翻译自检 | A | 0.5 h |
| Day 5 | ESM-3 GFP cookbook | **C** | 2 h |
| Day 5 | 批量 ESMFold pLDDT 过滤 | **C** | 1 h |

> **总开销估算**：CPU 60 h + V100 20 h + A100 5 h ≈ **¥160**，全队充 200 元够。

---

## 3. 持久存储（关键，决定你下次开机要不要重装）

| 路径 | 性质 | 用途 |
| --- | --- | --- |
| `/personal/` | **持久**，机型切换都在 | **所有 git clone、模型权重、嵌入缓存放这** |
| `/bohr/<dataset>/v1/` | **挂载只读** | 比赛数据集（GFP_data.xlsx、Exclusion_List 等），不要 cp 出来浪费空间 |
| `../2026Protein Design/`（本地） | **本地副本** | 本机开发用的赛事数据，与 `ProteinDesign/` 仓库平级，不进 Git |
| `/data/` 或 `/root/` | **临时**，关机即丢 | 别放重要东西 |
| `/tmp/` | 临时 | 跑完就清掉 |

**所有人遵守约定**：

```
/personal/
└── proteindesign/
    ├── ProteinDesign/        ← 你的 GitHub 仓库（git clone 来的）
    ├── third_party/          ← ProteinMPNN/ThermoMPNN/ESM 等第三方
    ├── inputs/pdb/           ← 下载的 PDB
    ├── outputs/              ← 跑出来的中间结果
    └── cache/                ← ESM 模型权重缓存
```

---

## 4. 第一次开实例：跑 init 脚本（5–10 分钟）

进入 Notebook 后，新建一个 cell（或开 Terminal），跑：

```bash
cd /personal && mkdir -p proteindesign && cd proteindesign

# 1) Clone 主仓库
git clone https://github.com/LynnChrisleetop/ProteinDesign.git
cd ProteinDesign

# 2) 一键安装所有依赖与第三方工具
bash scripts/bohrium_init.sh
```

跑完会得到：

- ✅ PyTorch / fair-esm / transformers / lightgbm / FLEXS / DnaChisel 全装好
- ✅ ProteinMPNN / ThermoMPNN / ColabDesign / ESM-3 仓库 clone 到 `third_party/`
- ✅ sfGFP `2B3P.pdb` 下载到 `inputs/pdb/`
- ✅ 自检：打印 GPU 是否可用、ESM 模型能否加载

---

## 5. 之后每次开实例（< 30 秒就能开工）

```bash
cd /personal/proteindesign/ProteinDesign
git pull origin main
# 直接开始跑实验
```

> 因为环境装在 `/personal/`，**机型切换 / 节点重启都不用重装**，只要镜像还是同一个。

---

## 6. 实验做完：把结果同步回 GitHub

```bash
cd /personal/proteindesign/ProteinDesign

# 只 push 真正重要的小文件
git add 参赛指南.md docs/ src/ outputs/*.csv outputs/*.json
git commit -m "expt: <做了什么>"
git push
```

> **大文件**（`*.pt`、`*.pdb`、`*.npz`、ESM 缓存）已经被 `.gitignore` 排除，不会被 push。

---

## 7. ⚠️ 关机！关机！关机！（重要 ×3）

Bohrium 按"实例运行时长"扣费，**不关机一直在烧钱**！

```
做完实验 → Notebook 右上角 Stop（或左侧 Stop Instance）
```

进阶：在创建实例时，把"最大运行时长"设成 **4 小时** 或 **8 小时**，到点自动停。

> 个人空间 `/personal/` 不会因为 Stop 丢失，下次 Start 就能继续。

---

## 8. 团队协作约定

| 资产 | 怎么共享 |
| --- | --- |
| **代码 / 配置 / submission.csv / 设计文档** | **GitHub `LynnChrisleetop/ProteinDesign`** |
| **大文件**（PDB、模型权重、嵌入 .npz） | **不进 Git**；放 Bohrium `/personal/` |
| **算力** | 队长充值，每人开自己的实例（Bohrium 计费按实例不按账号） |
| **训练好的 RF 模型 (.joblib)** | 小的（< 50 MB）可以 Git LFS；大的丢 Hugging Face Hub 或 Bohrium 共享目录 |
| **实验日志** | Weights & Biases（免费团队空间）或 `outputs/logs/` 进 Git |

**典型一天的协作流**：

```
[本地]                 [GitHub]                 [Bohrium]
  写代码 -- push -->  ProteinDesign  -- pull --> Notebook 实例
                                                  |
                                                跑实验
                                                  |
                                                push 结果
                                                  v
  pull <-- pull -- ProteinDesign  <-- push --   /
```

---

## 9. 常见问题

**Q1：第一次开 GPU 节点等了 5 分钟还没好？**
A：A100 高峰期会排队（早 9–11 点、晚 8–10 点），换 V100 或挑非高峰时段。

**Q2：`pip install` 太慢 / 超时？**
A：Bohrium 国内网络默认就装清华源，无需配置；如果发现还是慢，加参数：
```bash
pip install xxx -i https://pypi.tuna.tsinghua.edu.cn/simple
```
或者用 Bohrium 自带的 `LBG`（每个镜像默认装的镜像加速器）。

**Q3：`git clone` 卡住 / 失败？**
A：换 SSH 或加镜像：
```bash
# GitHub 镜像（国内访问加速）
git clone https://ghproxy.com/https://github.com/dauparas/ProteinMPNN.git
# 或者
git clone https://hub.fastgit.xyz/dauparas/ProteinMPNN.git
```

**Q4：模型权重下载（fair-esm、HuggingFace）失败？**
A：`bohrium_init.sh` 已经预设了 `HF_ENDPOINT=https://hf-mirror.com` 和 `TORCH_HOME=/personal/proteindesign/cache/torch`。如果还失败，看 `init.log` 里的具体报错。

**Q5：换机型后 `import torch` 报错说找不到 CUDA？**
A：你可能从 GPU 节点切回了 CPU 节点。`torch.cuda.is_available()` 自然返回 False，**代码里要写自适应**：
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```
我们的 src 脚手架里所有地方都已经这么写了。

**Q6：`/personal/` 满了？**
A：检查谁占空间最大：
```bash
du -h --max-depth=2 /personal/proteindesign | sort -h | tail -20
```
通常是 ESM 模型缓存（~/.cache/torch/hub）或 ESM-3 权重（几十 GB）。删掉用不到的：
```bash
rm -rf /personal/proteindesign/cache/torch/hub/checkpoints/esm2_t33_650M_UR50D.pt
```

**Q7：ESM-3 要 token，怎么搞？**
A：去 [forge.evolutionaryscale.ai](https://forge.evolutionaryscale.ai) 注册（免费学术档），拿到 API token 后：
```bash
mkdir -p ~/.esm
echo "你的token" > ~/.esm/api_token
```
之后 ESM-3 SDK 会自动读取。

**Q8：实例 Stop 后重启，环境还在吗？**
A：**`/personal/` 永远在**。但 `/data/`、`/tmp/`、`pip` 全局装的包（不在 `/personal/`）会丢。所以本手册推荐**所有第三方仓库和 cache 都放 `/personal/`**。

**Q9：跑到一半 GPU OOM（Out Of Memory）？**
A：把 batch size 调小（教程里 `BATCH_SIZE = 16` → 改 `8` 或 `4`），或换更大显存机型（V100_32g → A100_40g）。代码里也可以加 `torch.cuda.empty_cache()`。

**Q10：`jupyter` 卡住 / 内核崩溃？**
A：左侧 `Kernel → Restart Kernel`，或者直接在 Terminal 跑 `python src/xxx.py`，避免 Notebook 大对象占内存。

---

## 10. 安全 & 防呆

- ❌ **永远不要把 ESM-3 API token、HuggingFace token、个人邮箱密码 push 到 Git**。它们应该写在 `~/.esm/`、`~/.huggingface/` 这些 home 目录下，不会被 Git 追踪。
- ❌ **永远不要把 `Exclusion_List.csv` 之外的赛事数据 push 到公开仓库**。组织方把数据放在 `/bohr/...` 是为了限定访问，外泄会被取消资格。
- ✅ 本仓库的 `.gitignore` 已经把 `data/raw/`、`third_party/`、模型权重、PDB 等都排除了，正常用就不会泄漏。
- ✅ 提交比赛前**清空 `outputs/` 里的临时实验文件**，只保留最终 `submission.csv`，让仓库整洁。

---

## 11. 自检（每次 push 前）

```bash
# 在 ProteinDesign 根目录跑
python -m src.utils.validate_submission outputs/submission.csv
```

输出应当全 `[OK]`。如果出现 `[FAIL]`，**绝对不要提交**，按提示修复。详见 §7 自检清单（在 `参赛指南.md`）。

---

## 12. 一图流总览

```
┌────────────────────────────────────────────────────────────────┐
│  本地（Windows / Mac）                                          │
│  Cursor 编辑代码 + 改 markdown                                  │
│         │                                                       │
│         │ git push                                              │
│         v                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  GitHub: LynnChrisleetop/ProteinDesign               │       │
│  │  分支: main                                          │       │
│  │  内容: 代码、参赛指南、submission.csv、设计文档      │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                       │
│         │ git pull                                              │
│         v                                                       │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  Bohrium Notebook  ubuntu:22.04-py3.10-cuda12.1      │       │
│  │  /personal/proteindesign/                            │       │
│  │  ├── ProteinDesign/        (git 仓库)                │       │
│  │  ├── third_party/          (ProteinMPNN/ThermoMPNN..)│       │
│  │  ├── inputs/pdb/           (sfGFP 等 PDB)            │       │
│  │  ├── outputs/              (跑出来的中间结果)         │       │
│  │  └── cache/                (ESM 权重、HF cache)      │       │
│  │                                                       │       │
│  │  按需切机型：c4_m8_cpu / V100_32g / A100_40g         │       │
│  │  实验做完 → 立刻 Stop！                               │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                       │
│         │ git push 结果                                         │
│         v                                                       │
│  GitHub 同步给所有队员                                          │
└────────────────────────────────────────────────────────────────┘
```

---

## 附录：本文档版本

- v1.0 2026-05-14：初版
