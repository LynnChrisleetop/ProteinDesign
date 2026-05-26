"""把 outputs/seeds.csv 转成赛方提交格式 outputs/submission.csv。

赛方模板（CRLF 行尾，**不可改动**）：
    Team_Name,Seq_ID,Sequence
    <team>,1,<seq>
    ...

用法：
    python src/06_make_submission.py --team <YourTeamName>
    python src/06_make_submission.py --team <YourTeamName> --strict-check

--strict-check  跑前重做一次合规自检（长度/M/AA/Exclusion），失败则拒绝写出。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (  # noqa: E402
    EXCLUSION_CSV,
    OUTPUTS_DIR,
    SUBMISSION_TEMPLATE,
    check_sequence,
)


def strict_check(rows: list[dict], excl_set: set[str]) -> list[str]:
    errs: list[str] = []
    if len(rows) != 6:
        errs.append(f"必须恰好 6 条，当前 {len(rows)}")
    ids_seen = set()
    for r in rows:
        sid = int(r["Seq_ID"])
        seq = str(r["Sequence"]).strip().upper()
        chk = check_sequence(seq)
        if not chk["passes_all"]:
            errs.append(
                f"Seq_{sid}: 基础合规失败 (len={chk['length']}, "
                f"M={chk['starts_with_M']}, AA_ok={chk['only_standard_aa']})"
            )
        if seq in excl_set:
            errs.append(f"Seq_{sid}: 命中 Exclusion_List（exact match）")
        if sid in ids_seen:
            errs.append(f"Seq_ID 重复: {sid}")
        ids_seen.add(sid)
        if not (1 <= sid <= 6):
            errs.append(f"Seq_ID 越界: {sid}")
    if ids_seen != set(range(1, 7)):
        errs.append(f"Seq_ID 必须是 1..6，当前={sorted(ids_seen)}")
    return errs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--team", required=True, help="赛队名称（出现在 Team_Name 列）")
    p.add_argument("--seeds", default=str(OUTPUTS_DIR / "seeds.csv"))
    p.add_argument("--out", default=str(OUTPUTS_DIR / "submission.csv"))
    p.add_argument("--strict-check", action="store_true",
                   help="写出前重做一次合规 + Exclusion 自检")
    args = p.parse_args()

    print(f"[1] 读 {args.seeds}")
    seeds = pd.read_csv(args.seeds).dropna(subset=["Sequence"])
    seeds = seeds.sort_values("Seq_ID").reset_index(drop=True)
    print(f"  loaded {len(seeds)} seeds: Seq_IDs={seeds['Seq_ID'].astype(int).tolist()}")

    rows = []
    for _, r in seeds.iterrows():
        rows.append({
            "Team_Name": args.team,
            "Seq_ID": int(r["Seq_ID"]),
            "Sequence": str(r["Sequence"]).strip().upper(),
        })

    if args.strict_check:
        print("\n[2] 严格自检 ...")
        excl_df = pd.read_csv(EXCLUSION_CSV)
        excl_set = set(excl_df[excl_df.columns[0]].astype(str).str.strip().str.upper().tolist())
        errs = strict_check(rows, excl_set)
        if errs:
            print("  [FAIL] 自检不通过：")
            for e in errs:
                print(f"    - {e}")
            return 2
        print("  [ok] 全部 6 条通过自检")

    print(f"\n[3] 写出 {args.out} (CRLF 行尾，UTF-8 无 BOM) ...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Team_Name", "Seq_ID", "Sequence"],
                           lineterminator="\r\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    size = out_path.stat().st_size
    print(f"  -> {out_path}  ({size} bytes)")

    print("\n[4] 预览：")
    with out_path.open("rb") as f:
        head = f.read(200)
    print("  前 200 字节 (repr):", repr(head[:200]))
    print()
    for r in rows:
        print(f"  Seq_{r['Seq_ID']}  len={len(r['Sequence']):>3}  "
              f"start='{r['Sequence'][:8]}...'")

    if SUBMISSION_TEMPLATE and SUBMISSION_TEMPLATE.is_file():
        print("\n[5] 与赛方模板对比表头：")
        tmpl_head = SUBMISSION_TEMPLATE.open("rb").read(20)
        ours_head = out_path.open("rb").read(20)
        print(f"  模板  : {tmpl_head!r}")
        print(f"  我们的: {ours_head!r}")
        if tmpl_head[:18] == ours_head[:18]:
            print("  [ok] 表头字节级一致")
        else:
            print("  [WARN] 表头差异，请目检")

    print(f"\n[done] submission.csv 已生成，可直接提交。")
    print(f"       团队名='{args.team}'。如需改，重跑 --team <新名>。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
