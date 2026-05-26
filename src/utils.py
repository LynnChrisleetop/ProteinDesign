"""通用工具：路径解析、FASTA 解析、序列合规校验、IO 助手。

所有 src/ 下脚本都从这里 import 路径常量，避免硬编码。

数据目录解析优先级：
  1. 环境变量 GFP_DATA_DIR
  2. Bohrium 挂载点 /bohr/2025proteindesign-iw1n/v1/
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


def apply_mutations(wt: str, mut_str: str) -> str | None:
    """把 `A12B:C34D` 应用到 WT 序列。失败返回 None。"""
    parsed = parse_mutation_str(mut_str)
    if parsed is None:
        return None
    seq = list(wt)
    for orig, pos, new in parsed:
        idx = pos - 1
        if not 0 <= idx < len(seq):
            return None
        seq[idx] = new
    return "".join(seq)


def write_fasta(seqs: Dict[str, str], path: Path | str) -> None:
    out = []
    for name, seq in seqs.items():
        out.append(f">{name}")
        for i in range(0, len(seq), 60):
            out.append(seq[i:i + 60])
    Path(path).write_text("\n".join(out) + "\n", encoding="utf-8")
