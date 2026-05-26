#!/usr/bin/env bash
# =============================================================================
# bohrium_init.sh
# Bohrium Notebook 一键初始化脚本（GFP Protein Design 比赛）
# 镜像: ubuntu:22.04-py3.10-cuda12.1
# 用法: bash scripts/bohrium_init.sh   (在 ProteinDesign/ 仓库根目录跑)
# =============================================================================
set -euo pipefail

# --- 0. 基本路径 ---
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORKSPACE="${WORKSPACE:-/personal/biosys}"          # 持久工作区
THIRD_PARTY="${PROJECT_ROOT}/third_party"
INPUTS_DIR="${PROJECT_ROOT}/inputs"
OUTPUTS_DIR="${PROJECT_ROOT}/outputs"
CACHE_DIR="${WORKSPACE}/cache"
LOG_FILE="${PROJECT_ROOT}/init.log"

mkdir -p "${THIRD_PARTY}" "${INPUTS_DIR}/pdb" "${OUTPUTS_DIR}" \
         "${CACHE_DIR}/torch" "${CACHE_DIR}/huggingface"

# 把所有输出同时写到 init.log，方便排查
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "==============================================================="
echo "[bohrium_init] $(date '+%F %T')"
echo "Project root: ${PROJECT_ROOT}"
echo "Workspace:    ${WORKSPACE}"
echo "==============================================================="

# --- 1. 国内镜像加速（pip / huggingface / torch hub） ---
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TORCH_HOME="${CACHE_DIR}/torch"
export HF_HOME="${CACHE_DIR}/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}"

# 把这些环境变量持久化到 ~/.bashrc，下次 Stop->Start 不用重设
if ! grep -q "## proteindesign env" "${HOME}/.bashrc" 2>/dev/null; then
  cat >> "${HOME}/.bashrc" <<EOF

## proteindesign env (auto-added by bohrium_init.sh)
export PIP_INDEX_URL=${PIP_INDEX_URL}
export HF_ENDPOINT=${HF_ENDPOINT}
export TORCH_HOME=${TORCH_HOME}
export HF_HOME=${HF_HOME}
export TRANSFORMERS_CACHE=${HF_HOME}
EOF
  echo "[ok] 环境变量已写入 ~/.bashrc"
fi

# --- 2. 检查 Python / GPU ---
echo "--- Python ---"
python --version
echo "--- GPU ---"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | head -n 15
else
  echo "[warn] 当前节点无 GPU（CPU 节点也 OK，跑代码调试用）"
fi

# --- 3. 安装 Python 依赖 ---
echo ""
echo "[step 3/6] 安装核心 Python 依赖（5-10 分钟）..."
python -m pip install --upgrade pip wheel setuptools

# 3.1 PyTorch（CUDA 12.1 对应 torch 2.3+）
python -m pip install --upgrade \
    "torch>=2.1" "torchvision" \
    --index-url https://download.pytorch.org/whl/cu121 \
    || python -m pip install --upgrade "torch>=2.1"

# 3.2 蛋白学相关
python -m pip install --upgrade \
    fair-esm \
    transformers sentencepiece accelerate \
    biopython biopandas \
    "scikit-learn>=1.3" lightgbm xgboost optuna \
    pandas numpy openpyxl tqdm pyyaml joblib \
    matplotlib seaborn logomaker py3Dmol \
    flexs

# 3.3 DnaChisel（优先用本地版，否则装 pypi 版）
LOCAL_DNACHISEL="${PROJECT_ROOT}/../tools/DnaChisel"
if [ -d "${LOCAL_DNACHISEL}" ]; then
  python -m pip install -e "${LOCAL_DNACHISEL}"
elif [ -d "${WORKSPACE}/tools/DnaChisel" ]; then
  python -m pip install -e "${WORKSPACE}/tools/DnaChisel"
else
  python -m pip install dnachisel
fi

echo "[ok] Python 依赖安装完成"

# --- 4. Clone 第三方仓库 ---
echo ""
echo "[step 4/6] Clone 第三方仓库到 third_party/ ..."

clone_or_update() {
  local repo_url="$1"
  local dir_name="$2"
  local commit="${3:-}"
  local target="${THIRD_PARTY}/${dir_name}"
  if [ -d "${target}/.git" ]; then
    echo "  [skip] ${dir_name} 已存在（git pull 更新）"
    (cd "${target}" && git pull --ff-only) || true
  else
    echo "  [clone] ${repo_url} -> ${dir_name}"
    git clone --depth 1 "${repo_url}" "${target}" || \
      git clone "https://ghproxy.com/${repo_url}" "${target}"   # 失败则走镜像
  fi
  if [ -n "${commit}" ] && [ -d "${target}/.git" ]; then
    (cd "${target}" && git fetch --depth 50 origin "${commit}" 2>/dev/null && git checkout "${commit}") || true
  fi
}

clone_or_update https://github.com/dauparas/ProteinMPNN.git ProteinMPNN
clone_or_update https://github.com/Kuhlman-Lab/ThermoMPNN.git ThermoMPNN
clone_or_update https://github.com/sokrypton/ColabDesign.git ColabDesign
clone_or_update https://github.com/evolutionaryscale/esm.git esm3_repo
clone_or_update https://github.com/Edinburgh-Genome-Foundry/DnaChisel.git DnaChisel_upstream
# LigandMPNN 可选（占空间多，默认不装）
# clone_or_update https://github.com/dauparas/LigandMPNN.git LigandMPNN

# 装 ESM-3 SDK（需要本地 -e 安装）
if [ -d "${THIRD_PARTY}/esm3_repo" ]; then
  python -m pip install -e "${THIRD_PARTY}/esm3_repo" || true
fi

echo "[ok] 第三方仓库就绪"

# --- 5. 下载常用 PDB（sfGFP 2B3P 等）---
echo ""
echo "[step 5/6] 下载参考 PDB ..."

cd "${INPUTS_DIR}/pdb"
declare -A PDBS=(
  [2B3P]="sfGFP"
  [2WUR]="avGFP"
  [7LG4]="amacGFP"
  [2HPW]="cgreGFP"
  [2G6X]="ppluGFP"
)
for pdb in "${!PDBS[@]}"; do
  if [ ! -f "${pdb}.pdb" ]; then
    echo "  [download] ${pdb} (${PDBS[$pdb]})"
    wget -q "https://files.rcsb.org/download/${pdb}.pdb" || \
      curl -sLO "https://files.rcsb.org/download/${pdb}.pdb" || \
      echo "  [warn] ${pdb}.pdb 下载失败，可手动重试"
  else
    echo "  [skip] ${pdb}.pdb 已存在"
  fi
done
cd "${PROJECT_ROOT}"
echo "[ok] PDB 就绪"

# --- 6. 自检 ---
echo ""
echo "[step 6/6] 环境自检 ..."

python - <<'PY'
import sys, importlib
mods = ["torch", "esm", "transformers", "sklearn", "lightgbm",
        "Bio", "pandas", "numpy", "dnachisel"]
print(f"Python: {sys.version.split()[0]}")
for m in mods:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "?")
        print(f"  [ok]   {m:<14} {ver}")
    except Exception as e:
        print(f"  [FAIL] {m:<14} -> {e}")

import torch
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  mem:    {torch.cuda.get_device_properties(0).total_memory/(1024**3):.1f} GB")

# 测试 fair-esm 能否加载（只下载最小模型 esm2_t6_8M）
try:
    import esm
    print("\n[test] 加载 ESM2 8M（首次会下载约 30 MB）...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    print(f"  [ok] ESM2 8M loaded, embed_dim={model.embed_dim}")
except Exception as e:
    print(f"  [warn] ESM 加载失败: {e}")
PY

echo ""
echo "==============================================================="
echo "✅ Bohrium 环境初始化完成！"
echo ""
echo "下一步："
echo "  1. 编辑代码并 git push"
echo "  2. 在 Notebook 里跑: cd ${PROJECT_ROOT} && jupyter notebook"
echo "  3. 跑实验: python src/00_load_and_clean.py"
echo "  4. 实验做完立刻 Stop 实例！"
echo ""
echo "完整日志: ${LOG_FILE}"
echo "==============================================================="
