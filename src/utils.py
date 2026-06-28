"""通用工具：路径解析、FASTA 解析、序列合规校验、IO 助手。

所有 src/ 下脚本都从这里 import 路径常量，避免硬编码。

数据目录解析优先级：
  1. 环境变量 GFP_DATA_DIR
  2. 可选挂载点 /bohr/2025proteindesign-iw1n/v1/（云环境）
  3. 仓库同级目录 ../2026Protein Design/
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
LEN_MIN, LEN_MAX = 220, 250


def resolve_data_dir() -> Path:
    """按优先级返回赛事数据目录。找不到时抛 FileNotFoundError。"""
    candidates = []
    env = os.environ.get("GFP_DATA_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(Path("/bohr/2025proteindesign-iw1n/v1"))
    candidates.append(REPO_ROOT.parent / "2026Protein Design")
    for c in candidates:
        if c.is_dir() and (c / "GFP_data.xlsx").is_file():
            return c
    raise FileNotFoundError(
        "找不到赛事数据目录。请确认下列任一可用：\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\n或设置环境变量 GFP_DATA_DIR 指向含 GFP_data.xlsx 的目录。"
    )


DATA_DIR = resolve_data_dir() if (
    Path("/bohr/2025proteindesign-iw1n/v1").is_dir()
    or (REPO_ROOT.parent / "2026Protein Design").is_dir()
    or os.environ.get("GFP_DATA_DIR")
) else None

GFP_DATA_XLSX = DATA_DIR / "GFP_data.xlsx" if DATA_DIR else None
EXCLUSION_CSV = DATA_DIR / "Exclusion_List.csv" if DATA_DIR else None
WT_FASTA_TXT = DATA_DIR / "AAseqs of 5 GFP proteins_20260511.txt" if DATA_DIR else None
SUBMISSION_TEMPLATE = DATA_DIR / "submission_template.csv" if DATA_DIR else None

OUTPUTS_DIR = REPO_ROOT / "outputs"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
OUTPUTS_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def parse_wt_fasta(path: Path | str) -> Dict[str, str]:
    """解析赛方 5 条 WT 的 FASTA-like 文件。

    格式样例：
        >sfGFP
        MSKGEELFTG...
        # recommend PDB: 2B3P
    返回 {name: sequence}（已大写、去空白）。
    """
    text = Path(path).read_text(encoding="utf-8")
    seqs: Dict[str, str] = {}
    name: str | None = None
    buf: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf).upper()
            name = s[1:].split()[0]
            buf = []
        else:
            buf.append(re.sub(r"\s+", "", s))
    if name is not None:
        seqs[name] = "".join(buf).upper()
    return seqs


def check_sequence(seq: str) -> Dict[str, object]:
    """对一条序列做赛事合规检查。返回详细诊断字典。"""
    seq = seq.strip().upper()
    n = len(seq)
    bad_chars = sorted(set(seq) - VALID_AA)
    return {
        "length": n,
        "length_ok": LEN_MIN <= n <= LEN_MAX,
        "starts_with_M": seq.startswith("M"),
        "only_standard_aa": not bad_chars,
        "bad_chars": bad_chars,
        "passes_all": (
            LEN_MIN <= n <= LEN_MAX
            and seq.startswith("M")
            and not bad_chars
        ),
    }


def fmt_check(check: Dict[str, object]) -> str:
    flags = []
    flags.append("len" if check["length_ok"] else f"LEN={check['length']}!")
    flags.append("M" if check["starts_with_M"] else "NoM!")
    flags.append("AA" if check["only_standard_aa"] else f"BAD={check['bad_chars']}")
    return " | ".join(flags)


_MUT_RE = re.compile(r"^([A-Z])(\d+)([A-Z*.])$", re.IGNORECASE)

# 两种编号体系（**真实编号体系经过实测确认**）：
#   "with_M"  : 1-based 含起始 M（M=1, S=2, ...）；文献 / 赛方 FASTA / winner_diff
#   "skip_M"  : 1-based 跳过起始 M（S=1, K=2, ...）；Sarkisyan GFP_data.xlsx
#
# 实测：avGFP 数据中的 'A109D' → orig='A' 实际是 av[109]（Python 0-based），
#       而不是 av[108]，即数据集用 "skip_M" 编号。文献的 "S65T" → S 是 av[64]，
#       即文献用 "with_M" 编号。两套体系**相差 1 位**。

MUTATION_NUMBERING_WITH_M = "with_M"   # 默认：所有文献 / 赛方 / 设计 / WT FASTA
MUTATION_NUMBERING_SKIP_M = "skip_M"   # Sarkisyan 数据集专用


def parse_mutation_str(mut_str: str) -> list[tuple[str, int, str]] | None:
    """解析 `A12B:C34D` 这种突变描述串。返回 [(orig, 1-based_pos, new), ...]。

    特殊取值：
      - 'WT' → 返回 []
      - 含 `*`（终止子） → 返回 None（视为无效）
      - `.` (no-op) → 视为同义，跳过该条
      - 无法解析的子项 → 跳过
    """
    if not isinstance(mut_str, str):
        return None
    s = mut_str.strip()
    if s.upper() == "WT" or s == "":
        return []
    out: list[tuple[str, int, str]] = []
    for tok in s.split(":"):
        tok = tok.strip()
        m = _MUT_RE.match(tok)
        if not m:
            continue
        orig, pos, new = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
        if new == "*":
            return None
        if new == ".":
            continue
        out.append((orig, pos, new))
    return out


def _pos_to_idx(pos: int, numbering: str) -> int:
    """把 1-based pos 按编号体系映射到 Python 0-based index。"""
    if numbering == MUTATION_NUMBERING_WITH_M:
        return pos - 1
    if numbering == MUTATION_NUMBERING_SKIP_M:
        return pos        # skip M 起始 M=index 0，pos 1 对应 index 1
    raise ValueError(f"unknown numbering: {numbering!r}")


def apply_mutations(
    wt: str,
    mut_str: str,
    numbering: str = MUTATION_NUMBERING_WITH_M,
    strict: bool = False,
) -> str | None:
    """把 `A12B:C34D` 应用到 WT 序列。失败返回 None。

    Args:
        wt: WT AA 序列（含起始 M）
        mut_str: 突变串
        numbering: "with_M"（文献，默认）或 "skip_M"（Sarkisyan 数据集）
        strict: 若 True，则当 orig AA 与 WT 实际不符时返回 None；
                若 False（默认），仅做替换不校验 orig（兼容旧行为）

    返回突变后序列字符串，或解析失败时 None。
    """
    parsed = parse_mutation_str(mut_str)
    if parsed is None:
        return None
    seq = list(wt)
    for orig, pos, new in parsed:
        idx = _pos_to_idx(pos, numbering)
        if not 0 <= idx < len(seq):
            return None
        if strict and seq[idx] != orig:
            return None
        seq[idx] = new
    return "".join(seq)


def detect_numbering(wt_map: dict[str, str], sample_muts: list[tuple[str, str]]) -> str:
    """通过抽样匹配，自动判断突变串使用的编号体系。

    sample_muts: [(gfp_type, mut_str), ...]，至少 ≥ 20 条避免巧合。
    """
    score = {MUTATION_NUMBERING_WITH_M: 0, MUTATION_NUMBERING_SKIP_M: 0}
    n_tested = 0
    for gfp, mut in sample_muts:
        wt = wt_map.get(gfp)
        if wt is None:
            continue
        parsed = parse_mutation_str(mut)
        if not parsed:
            continue
        for orig, pos, _ in parsed:
            for sys_name in score:
                idx = _pos_to_idx(pos, sys_name)
                if 0 <= idx < len(wt) and wt[idx] == orig:
                    score[sys_name] += 1
            n_tested += 1
    if n_tested == 0:
        raise RuntimeError("no testable mutations")
    best = max(score, key=score.get)
    confidence = score[best] / max(n_tested, 1)
    if confidence < 0.95:
        raise RuntimeError(
            f"numbering detection ambiguous: scores={score} of {n_tested} tested"
        )
    return best


def write_fasta(seqs: Dict[str, str], path: Path | str) -> None:
    out = []
    for name, seq in seqs.items():
        out.append(f">{name}")
        for i in range(0, len(seq), 60):
            out.append(seq[i:i + 60])
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
