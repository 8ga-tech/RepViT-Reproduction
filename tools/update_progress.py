# -*- coding: utf-8 -*-
"""tools/update_progress.py —— 就地更新 PROGRESS.md 的模块状态表（规格书 §4.2(d)）。

用法：
    python tools/update_progress.py --module M06 --status done --note "val_macro_f1=0.9xx"

行为契约（冻结）：
  * 只在**已有的那张模块状态表**里改，定位方式是正则匹配表格行 `| M06 | ... |`；
  * 找不到模块行 -> 报错并 exit(1)，绝不静默新建第二张表（静默建表 = 进度被记到别处，
    验收时看不到，是最容易骗过自己也骗过评委的一种错）；
  * 改三处：该行的「状态」列、该行的备注、顶部的 `last_update`（ISO8601 带时区）
    与 `overall`（`N/13 done, M blocked`，N/M 由表格里所有模块行现算，不手写）；
  * 写回 UTF-8，换行统一为 \\n。

备注写法：追加到该行最后一个非空单元格，格式为 `备注:<text>`；重复执行时会先删掉旧的
`备注:` 片段再写新的，因此脚本是幂等的（不会一行里堆十个备注）。
"""
from __future__ import annotations

import argparse
import datetime
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

NOTE_MARK = "备注:"
MODULE_RE = re.compile(r"^M\d+$")


def _resolve(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def split_row(line: str) -> list[str]:
    """'| M06 | x | y |' -> ['', 'M06', 'x', 'y', '']；保留首尾空段以便原样拼回。"""
    return [c.strip() for c in line.split("|")]


def join_row(cells: list[str]) -> str:
    """与 Markdown 表格的通行写法对齐：'|' + ' a | b |' -> '| a | b |'。"""
    inner = " | ".join(cells[1:-1])
    return f"| {inner} |"


def find_table(lines: list[str]) -> tuple[int, int, list[str]]:
    """定位模块状态表，返回 (表头行号, 表格结束行号(不含), 表头单元格)。

    只认「表头里同时出现『模块』和『状态』」的那张表，不靠固定行号（PROGRESS.md
    上方还有别的区块，硬编码行号一改格式就错位）。
    """
    for i, ln in enumerate(lines):
        if not ln.lstrip().startswith("|"):
            continue
        cells = split_row(ln)
        if "模块" in cells and "状态" in cells:
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            return i, j, cells
    raise LookupError("在 PROGRESS.md 里找不到「模块状态」表（表头需同时含『模块』与『状态』两列）")


def is_separator(cells: list[str]) -> bool:
    body = [c for c in cells[1:-1] if c]
    return bool(body) and all(set(c) <= set("-: ") for c in body)


def normalise_status(s: str) -> str:
    return re.sub(r"[\s_\-]+", "", s.strip().lower())


def is_done(status: str) -> bool:
    return normalise_status(status) in ("done", "完成", "已完成")


def is_blocked(status: str) -> bool:
    t = status.strip().lower()
    return ("blocked" in t) or ("阻塞" in t)


def main() -> int:
    ap = argparse.ArgumentParser("更新 PROGRESS.md 的模块状态表")
    ap.add_argument("--module", required=True, help="模块号，如 M06")
    ap.add_argument("--status", required=True,
                    help="新状态，如 done / in_progress / blocked / pending（自由文本，写入表格）")
    ap.add_argument("--note", default=None, help="备注，追加到该行末尾，格式 备注:<text>")
    ap.add_argument("--file", default="PROGRESS.md",
                    help="PROGRESS.md 路径（默认仓库根；改它只为自测时不污染正式文件）")
    args = ap.parse_args()

    module = args.module.strip()
    if not MODULE_RE.match(module):
        print(f"[ERROR] --module 必须是 M + 数字 的形式（如 M06），收到 {module!r}")
        return 1
    status = args.status.strip()
    if not status:
        print("[ERROR] --status 不能为空")
        return 1

    path = _resolve(args.file)
    if not path.exists():
        print(f"[ERROR] 找不到 {path}")
        return 1
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    try:
        head_i, end_i, header = find_table(lines)
    except LookupError as e:
        print(f"[ERROR] {e}；拒绝新建第二张表。")
        return 1

    col_module = header.index("模块")
    col_status = header.index("状态")
    # 备注写哪一列：优先表头里真的叫「备注」的列，其次「自检」列，都没有就写最后一列。
    # header 由 '| a | b |' 切出来，首尾各一个空段，所以「最后一列」的下标是 len-2。
    col_note = next((i for i, c in enumerate(header) if c == "备注"),
                    next((i for i, c in enumerate(header) if c == "自检"), len(header) - 2))

    target = None
    module_rows = 0
    for i in range(head_i + 1, end_i):
        cells = split_row(lines[i])
        if is_separator(cells):
            continue
        if col_module >= len(cells):
            continue
        if not MODULE_RE.match(cells[col_module]):
            continue
        module_rows += 1
        if cells[col_module] == module:
            target = i

    if target is None:
        found = []
        for i in range(head_i + 1, end_i):
            cells = split_row(lines[i])
            if col_module < len(cells) and MODULE_RE.match(cells[col_module]):
                found.append(cells[col_module])
        print(f"[ERROR] 模块状态表里找不到 {module}（表内现有：{found}）")
        print("        已按契约中止：不新建表格、不写任何内容，exit(1)。")
        return 1

    # ---- 1) 状态列 ----
    cells = split_row(lines[target])
    old_status = cells[col_status]
    cells[col_status] = status

    # ---- 2) 备注：追加到最后一个非空单元格（自检列），可重复执行 ----
    note_cell = None
    if args.note is not None:
        note = args.note.strip()
        note_cell = col_note
        while len(cells) <= note_cell:          # 行尾空单元格省略时补齐，保持列数不变
            cells.insert(-1, "")
        cur = re.sub(r"\s*" + re.escape(NOTE_MARK) + r"[^|]*$", "", cells[note_cell]).strip()
        cells[note_cell] = f"{cur} {NOTE_MARK}{note}".strip() if cur else f"{NOTE_MARK}{note}"

    lines[target] = join_row(cells)

    # ---- 3) 顶部字段：last_update / overall ----
    now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    done = blocked = 0
    for i in range(head_i + 1, end_i):
        c = split_row(lines[i])
        if is_separator(c) or col_status >= len(c):
            continue
        if not (col_module < len(c) and MODULE_RE.match(c[col_module])):
            continue
        st = c[col_status]
        if is_done(st):
            done += 1
        if is_blocked(st):
            blocked += 1

    re_last = re.compile(r"^(\s*-\s*last_update\s*:\s*).*$")
    re_overall = re.compile(r"^(\s*-\s*overall\s*:\s*)\d+\s*/\s*\d+\s+done\s*,\s*\d+\s+blocked(.*)$")
    re_overall_loose = re.compile(r"^(\s*-\s*overall\s*:\s*)(.*)$")
    got_last = got_overall = False
    for i, ln in enumerate(lines[:head_i]):
        if not got_last and re_last.match(ln):
            lines[i] = re_last.sub(rf"\g<1>{now}", ln)
            got_last = True
            continue
        if not got_overall:
            m = re_overall.match(ln)
            if m:
                lines[i] = f"{m.group(1)}{done}/{module_rows} done, {blocked} blocked{m.group(2)}"
                got_overall = True
            elif re_overall_loose.match(ln):
                print(f"[warn] overall 行格式与模板不一致，已按模板重写：{ln.strip()}")
                lines[i] = re_overall_loose.sub(
                    rf"\g<1>{done}/{module_rows} done, {blocked} blocked", ln)
                got_overall = True
    if not got_last:
        print("[warn] 顶部没有 `- last_update:` 行，未写入时间戳")
    if not got_overall:
        print("[warn] 顶部没有 `- overall:` 行，未写入总进度")

    out = "\n".join(lines) + "\n"            # 统一 \n，末尾保留一个换行
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)

    print("=" * 72)
    print(f" update_progress —— {path.as_posix()}")
    print(f" 模块 {module}：状态 '{old_status}' -> '{status}'")
    if args.note is not None:
        print(f"   备注已写入第 {note_cell + 1} 列（0-based 单元格 {note_cell}）：{NOTE_MARK}{args.note.strip()}")
    print(f" last_update = {now}")
    print(f" overall     = {done}/{module_rows} done, {blocked} blocked（由表内 {module_rows} 行现算）")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
