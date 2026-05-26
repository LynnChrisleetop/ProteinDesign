# 切 V100 一键开炮指南（Day 3+）

> 把"切完 GPU 之后到底先做什么"写死，避免烧着 GPU 计时器找命令。
> CPU 阶段（Day 1–2.5）的所有产物已经在 git 里，切到 GPU 实例 `git pull` 就能继续。

---

## 0. 切之前的最后检查（**还在 CPU 上做完**）

```bash
cd /personal/biosys/ProteinDesign
git status                                # 应该是 clean
ls outputs/seeds.csv outputs/submission.csv outputs/position_pool.csv
```

确认上面 3 个文件都在 → 安全切。

提交保底（防止 Day 3 跑炸了交不出来）：

```bash
# 已经做了；submission.csv 在 outputs/，CRLF + 模板对齐
head -2 outputs/submission.csv | xxd | head -2
```

---

## 1. 在 Bohrium 控制台切实例

1. 当前 CPU 实例 → **Stop**（注意不要 Terminate，工作区数据保留）
2. **Restart** 时改配置：
   - **机型**：`1xV100_32g` 或 `1xA100_40g`（首跑用 V100 即可，便宜）
   - **镜像**：保持 `ubuntu:22.04-py3.10-cuda12.1`（不要换镜像，不然依赖要重装）
   - **挂载**：勾上 `2025proteindesign-iw1n/v1`（不勾也行，本地已经有数据）
3. 实例起来后，VS Code/Cursor remote 重新连。
4. 终端里：
   ```bash
   nvidia-smi          # 看到 V100/A100 → 切成功
   ```

---

## 2. 重新装一遍 ML 依赖（首次切 GPU 实例必做）

CPU 实例上装过的包在新实例里**全没了**。Bohrium 工作区只保留 `/personal/biosys/` 下的代码和数据。

```bash
cd /personal/biosys/ProteinDesign
bash scripts/bohrium_init.sh
```

预期耗时：5–10 分钟。看到下面输出代表成功：

```
[ok] ESM2 8M loaded, embed_dim=320
CUDA available: True
  device: Tesla V100-SXM2-32GB
✅ Bohrium 环境初始化完成！
```

如果失败，查看 `init.log`。

---

## 3. Day 3.1：ESM 嵌入（**¥ 主要花费**）

### 3.1.1 烟测（5 min，先确保管线没问题）

```bash
python src/04_embed_esm.py --model t12_35M --batch 32 --limit 200
```

应该看到：
- `[device] cuda`
- `built 200 sequences`
- 输出 `outputs/esm_embeddings.npz` ≈ 1 MB

### 3.1.2 正式跑：35M 全数据（推荐，¥ 最低）

```bash
python src/04_embed_esm.py --model t12_35M --batch 64
```

- 数据量约 141k → V100 大概 15–25 分钟
- 输出 `outputs/esm_embeddings.npz` ≈ 250 MB
- 耗费 ≈ **¥3–5**

### 3.1.3 升级：150M（可选，效果更好，¥¥）

```bash
python src/04_embed_esm.py --model t30_150M --batch 16
```

- V100 ≈ 60 min
- 输出 `outputs/esm_embeddings.npz` ≈ 350 MB（D=640）
- 耗费 ≈ **¥10–15**

> **建议**：先 35M 跑通 04→05，模型能上 R² 0.35+ 再升 150M；否则先 ablation 调超参更划算。

---

## 4. Day 3.2：训练亮度回归模型（**CPU 也能跑，但有 GPU 更快**）

```bash
python src/05_train_regressor.py --model lgbm --tag esm35m_lgbm
```

预期：
- LightGBM 大概 5–10 分钟（500–2000 棵树）
- 看 `outputs/05_metrics_esm35m_lgbm.json` 的 `test.r2`
- **目标**：`test R² >= 0.40`（教程基线 0.28，超过即为 "强模型"）

如果 R² 不达标，做 ablation：

```bash
# 1. 试 RF 作基线
python src/05_train_regressor.py --model rf --tag esm35m_rf

# 2. 加 stratified by mutations 数（看少突变 vs 多突变误差）
# 3. 试 linear target 看是否 log scale 有损
python src/05_train_regressor.py --target linear --tag esm35m_lgbm_linear
```

---

## 5. Day 3.3：生成候选 + ML 筛选（待写：`src/06b_generate_candidates.py`）

> 这一步还没写，但思路明确，等 04/05 跑完再写更精准（因为要根据模型的解释做特征工程）：

1. **组合扫描**：基于 30 位点池（`outputs/position_pool.csv`），生成 1–6 突变组合，约 10k 条候选
2. **ML 评分**：用 04 嵌入 + 05 模型预测，留 Top-200
3. **合规过滤**：跑 `07_dnachisel_check.py` 流程，淘汰不合规
4. **多样性筛选**：保留 Hamming 互距 ≥ 5 的子集
5. **人工评审**：从 Top-50 选 2–3 条替换当前 Seq_4/5/6

---

## 6. Day 4：热稳定（ThermoMPNN）

```bash
# ThermoMPNN 仓库已经在 scripts/bohrium_init.sh 装到了 third_party/ThermoMPNN
ls third_party/ThermoMPNN/
ls inputs/pdb/                    # 2B3P.pdb (sfGFP), 2WUR.pdb (avGFP) ...
```

- 对 Top-200 候选做位点 ΔΔG 扫描
- 筛 Top-50 with ΔΔG > 0 的"稳定增益"
- 与 Day 3.3 的 ML 亮度分数加权排序

---

## 7. 结束 GPU（**很重要！！！**）

任何阶段结束都要：

1. `git add . && git commit -m "..."` → 保存代码 / 模型 / 结果
2. `git push`（如果用了私有仓库）
3. Bohrium 控制台 **Stop** 实例（不是 Terminate）
4. 检查计费页确认实例已停

V100 大约 ¥10–15 / 小时；A100 ¥20–30 / 小时。**每多挂 1 小时 = 1 杯奶茶**。

---

## 8. 预算预估

| 阶段 | 实例 | 时长 | 费用 |
|---|---|---|---|
| Day 3.1 烟测 | V100 | 10 min | ¥2 |
| Day 3.1 35M 全数据 | V100 | 25 min | ¥5 |
| Day 3.2 训练 + 调参 | V100 | 30 min | ¥6 |
| Day 3.3 候选生成 + ML 筛 | V100 | 30 min | ¥6 |
| Day 4 ThermoMPNN | V100 | 60 min | ¥12 |
| Day 4 ESMFold 验证 | V100 | 30 min | ¥6 |
| Day 3.1 升 150M（可选） | V100 | 60 min | ¥12 |
| **小计** | | **3–4 h** | **¥35–50** |

预留 50% buffer → 总预算 **¥60–80** 内可以做完。

---

## 9. 常见问题

**Q: nvidia-smi 显示有 GPU 但 torch.cuda.is_available() = False**
- 镜像不对，重启实例并选 `ubuntu:22.04-py3.10-cuda12.1`

**Q: ESM 模型下载卡住**
- `bohrium_init.sh` 已经把 HF_ENDPOINT 设为 hf-mirror.com，应当自动走国内镜像
- 仍失败：`huggingface-cli download facebook/esm2_t12_35M_UR50D`

**Q: pip install 卡在 CMake / dnachisel**
- 已经换为清华源（`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`），如果仍卡：`pip install dnachisel --no-build-isolation`

**Q: LightGBM out of memory**
- `--num-leaves 31 --feature-fraction 0.3` 调小，或对训练集随机降采样

---

## 10. 紧急情况

如果 GPU 实例跑挂、计费失控：
1. **立刻** Bohrium 控制台 **Stop** 当前实例
2. `outputs/submission.csv` 已经保底，赛事可以提交
3. 任何时候有 commit 推上 git，代码不会丢
