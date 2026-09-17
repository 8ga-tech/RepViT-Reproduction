# -*- coding: utf-8 -*-
"""证据指针校验器（t27 新增，随 selfcheck 的 `paths` 项一起跑）。

用途：校验 `report/REQUIREMENTS_AUDIT.md` 里**每一条证据引用**是否仍然可核对 ——
这类引用在 t2 写成，之后报告被压缩 / 违规表被删 / 章节重排，指针会集体失效；
本脚本把它变成**可机检**的闸，避免再出现「判定=满足但证据不存在」的复发。

校验两类指针：
  ① 产物/文件引用（反引号里的路径）：
     - 带目录的仓库相对路径 → 必须存在（tracked 或至少在磁盘上存在）；
     - 裸文件名（如 `bench.jsonl`）→ 必须能在 tracked 文件里唯一或部分解析；
     - 通配/占位（`*`、`<...>`、`...`、`..`）→ 记为 configurable，不判失败；
     - 首段不是仓库顶层项的（如上游仓库的 `model/repvit.py`）→ 记为外部引用，不判失败；
     - `_coord/**` → 仓库外的队内协调文件，存在即通过（记为 outside_repo）；
     - 同一句里显式声明「不存在/未落盘/无实跑」的 → 视为**如实披露**，不判失败。
  ② 章节号引用（`§x.y`）：
     - 先判定目标是**自指**（本文档自己的 §1~§12）还是**跨文档**（如 `report/REPORT.md` §4.3）：
       规则 = 该 § 前 40 字符内若出现被引文档名，则整段 §-run 归该文档，否则归本文档；
     - 再回到目标文档的**实际标题**里核对该编号存在（`### 4.3` / `## 五、` / `### 10.6` 都识别）。

用法：
    python tools/check_audit_evidence.py              # 人类可读摘要；有失败则 exit 1
    python tools/check_audit_evidence.py --json x.json
    python tools/check_audit_evidence.py --audit <其它审计文档>   # 负向验证用
    python tools/check_audit_evidence.py --quiet      # 只打印一行总结（selfcheck 用）

只读：本脚本只读取文件，不写仓库内任何产物。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
REPO = HERE.parents[1]
DEFAULT_AUDIT = REPO / "report" / "REQUIREMENTS_AUDIT.md"

# ---------------------------------------------------------------- 被引文档登记
# 文中出现的写法 → 仓库内路径（`_coord/...` 为仓库外协调文件，见 OUTSIDE_REPO_PREFIXES）
DOC_TOKENS = {
    "report/REPORT.md": "report/REPORT.md",
    "REPORT.md": "report/REPORT.md",
    "report/PPT_CONTENT.md": "report/PPT_CONTENT.md",
    "PPT_CONTENT.md": "report/PPT_CONTENT.md",
    "report/SPEC13_SWITCH_RUNBOOK.md": "report/SPEC13_SWITCH_RUNBOOK.md",
    "SPEC13_SWITCH_RUNBOOK.md": "report/SPEC13_SWITCH_RUNBOOK.md",
    "report/IMAGENETV2_PROVENANCE.md": "report/IMAGENETV2_PROVENANCE.md",
    "IMAGENETV2_PROVENANCE.md": "report/IMAGENETV2_PROVENANCE.md",
    "report/REQUIREMENTS_AUDIT.md": "report/REQUIREMENTS_AUDIT.md",
    "REQUIREMENTS_AUDIT.md": "report/REQUIREMENTS_AUDIT.md",
    "report/LOGITS_AUDIT_FINDINGS.md": "report/LOGITS_AUDIT_FINDINGS.md",
    "LOGITS_AUDIT_FINDINGS.md": "report/LOGITS_AUDIT_FINDINGS.md",
    "report/LOGITS_AUDIT.md": "report/LOGITS_AUDIT.md",
    "report/SYNC_NOTES.md": "report/SYNC_NOTES.md",
    "SYNC_NOTES.md": "report/SYNC_NOTES.md",
    "report/VERIFICATION_LOGITS_PPT.md": "report/VERIFICATION_LOGITS_PPT.md",
    "VERIFICATION_LOGITS_PPT.md": "report/VERIFICATION_LOGITS_PPT.md",
    "PROVENANCE.md": "PROVENANCE.md",
    "README.md": "README.md",
    "_coord/POLICY_IMAGENETV2_ONLY.md": "_coord/POLICY_IMAGENETV2_ONLY.md",
    "POLICY_IMAGENETV2_ONLY.md": "_coord/POLICY_IMAGENETV2_ONLY.md",
}
# 文档名匹配用（长 token 优先，避免 `REPORT.md` 吃掉 `report/REPORT.md`）
DOC_TOKEN_RE = re.compile("|".join(re.escape(k) for k in
                                  sorted(DOC_TOKENS, key=len, reverse=True)))

OUTSIDE_REPO_PREFIXES = ("_coord/",)          # 队内协调文件：仓库外、不入提交物
# 显式标注「上游 / 仓库外」的引用：如 `上游 THU-MIG/RepViT 的 model/repvit.py`、`仓库外（data/）`
EXTERNAL_MARKERS = ("上游", "仓库外", "非本仓库", "外部仓库")
ABSENCE_MARKERS = ("不存在", "未落盘", "无实跑", "缺失", "无该产物", "Test-Path` = False",
                   "不参与", "已删除", "不可用", "被 `.gitignore` 排除")
PATH_EXT = (".md", ".json", ".jsonl", ".csv", ".txt", ".py", ".sh", ".svg", ".yaml", ".yml",
            ".png", ".jpg", ".jpeg", ".docx", ".pdf", ".pptx", ".html", ".log", ".pt", ".onnx", ".xml")
# 扩展名白名单的**单一来源**（t38 F-2）：正文路径引用与「修复方案列」路径引用共用同一份，
# 避免两处白名单再漂移 —— 此前 PLAN_PATH_RE 漏了 `.svg/.pdf/.docx/.pptx/.log`，
# 于是 4 处指向**不存在 SVG** 的「可直接执行」指令逃过了自动闸。
EXT_ALT = "|".join(e.lstrip(".") for e in PATH_EXT)
PATH_RE = re.compile(
    r"`([A-Za-z0-9_./\-\u4e00-\u9fff*<>…]+?\.(?i:" + EXT_ALT + r"))"
    r"(?::([0-9\-–,，\s]+))?`")
SECTION_RE = re.compile(r"§\s?([0-9]+(?:\.[0-9]+)*|[一二三四五六七八九十]+)")
HEADING_RE = re.compile(r"^(#{1,4})\s+([0-9]+(?:\.[0-9]+)*|[一二三四五六七八九十]+)(?:[.、)）]|\s|$)")
VERDICT_RE = re.compile(r"^(满足|部分|缺失|挂起（待考核方清单）)$")
CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# ---- 可「现场重算并比对」的断言语法（t30 新增；文中按这套写法写，校验器才认）----
#   计数：`git ls-files onnx/` … **N 个**   → tracked onnx 文件数
#         官方 ONNX **N 个**                 → tracked 的 *_in1k.onnx（官方家族）数
#   体积：`path` … 实测 **N B**             → os.path.getsize
#   行数：`path` … 实测 **N 行**            → len(text.splitlines())
#   数据行：`path` … **N 行数据**           → CSV: 行数−1；Markdown 表: 表格行数−2
#   命中：`git grep -F -- "lit" [-- 路径]` → **N 命中**；`git grep -lE "pat"` → **N 个文件**；
#         `git grep -cE "pat"` → **N 行**（Σ 每文件计数）
COUNT_TOTAL_RE = re.compile(r"`git ls-files onnx/`[^\n]{0,30}?\*\*(\d+) 个\*\*")
COUNT_OFFICIAL_RE = re.compile(r"官方 ONNX\s?\*{0,2}(\d+) 个\*{0,2}")
SIZE_RE = re.compile(r"`([^`\n]+)`(?:(?!`)[^\n]){0,40}?实测 \*\*([\d,]+) B\*\*")
LINES_RE = re.compile(r"`([^`\n]+)`(?:(?!`)[^\n]){0,40}?实测 \*\*([\d,]+) 行\*\*")
DATAROWS_RE = re.compile(r"`([^`\n]+)`(?:(?!`)[^\n]){0,40}?\*\*([\d,]+) 行数据\*\*")
GREP_CMD_RE = re.compile(
    r"`git grep -(F|l|c) -- ([\"'])(.+?)\2((?: -- [^`]+)?)`"
    r"|`git grep -(lE|cE) ([\"'])(.+?)\6((?: -- [^`]+)?)`")
GREP_CLAIM_RE = re.compile(r"\*\*(\d+)\s*(命中|个文件|行)\*\*")
GREP_CLAIM_RE2 = re.compile(r"\*\*(\d+)\*\*\s*(命中|个文件|行)")
# 自指脚本无法断言自身：体积/行数断言若指向审计文档本身 → 跳过（记 INFO）
SELF_REF_FILES = ("report/REQUIREMENTS_AUDIT.md",)
# 自指触发器：出现这些词时，紧跟的 §号 指向本文档自己的章节（如「修复见 §9 得分项 G-5」）
SELF_TRIGGERS = ("见", "详见", "修复见", "参见", "按", "本报告", "本节", "本文档", "上表", "下表")
CLAUSE_DELIMS = "，。；：、（）()「」【】/|“”\"' \t"


def cn_to_int(s: str) -> int | None:
    """中文数字 → 整数（只覆盖本文档用到的 1~99 章号：十四 / 二十 / 一 …）。"""
    if not s:
        return None
    if s == "十":
        return 10
    total, section = 0, 0
    for ch in s:
        if ch == "十":
            section = (section or 1) * 10
            total += section
            section = 0
        elif ch in CN_DIGITS:
            section = CN_DIGITS[ch]
        else:
            return None
    return total + section


def id_variants(sid: str) -> set[str]:
    """编号的等价写法：`五` ↔ `5`；`十四` ↔ `14`。"""
    out = {sid}
    if sid.isdigit() or re.fullmatch(r"\d+(\.\d+)*", sid):
        return out
    n = cn_to_int(sid)
    if n is not None:
        out.add(str(n))
    return out


# ---------------------------------------------------------------- helpers
def _tracked_files(repo: pathlib.Path) -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=repo, capture_output=True,
                         text=True, encoding="utf-8").stdout
    return [x for x in out.splitlines() if x.strip()]


def heading_ids(path: pathlib.Path) -> set[str]:
    """markdown 标题编号集合（含中/阿拉伯等价写法）：`### 4.3 x` → {4.3}；`## 五、x` → {五, 5}。"""
    if not path.exists():
        return set()
    ids: set[str] = set()
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = HEADING_RE.match(ln.strip())
        if m:
            ids |= id_variants(m.group(2))
    return ids


def audit_headings(path: pathlib.Path) -> set[str]:
    return heading_ids(path)


def split_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


# ---------------------------------------------------------------- 主扫描
def scan(audit_path: pathlib.Path | None = None, repo: pathlib.Path | None = None) -> dict:
    audit_path = pathlib.Path(audit_path or DEFAULT_AUDIT)
    repo = pathlib.Path(repo or REPO)
    text = audit_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    tracked = _tracked_files(repo)
    by_base: dict[str, list[str]] = {}
    for f in tracked:
        by_base.setdefault(pathlib.Path(f).name, []).append(f)

    fail: list[dict] = []
    info: list[dict] = []
    n_rows = n_paths = n_secs = 0

    # ---- ① 证据列的路径引用（只查 满足/部分 行的证据列） ----
    for i, ln in enumerate(lines, 1):
        if not ln.startswith("|"):
            continue
        cells = split_cells(ln)
        if len(cells) < 4 or not VERDICT_RE.match(cells[2]) or cells[2] not in ("满足", "部分"):
            continue
        n_rows += 1
        ev = cells[1]
        for m in PATH_RE.finditer(ev):
            path, loc = m.group(1), (m.group(2) or "")
            n_paths += 1
            if any(ch in path for ch in "*<>…") or ".." in path:
                info.append(dict(kind="GLOB_OR_PLACEHOLDER", line=i, ref=path))
                continue
            head = path.split("/")[0]
            if head in OUTSIDE_REPO_PREFIXES:
                if (repo.parent / path).exists():
                    info.append(dict(kind="OUTSIDE_REPO_EXISTS", line=i, ref=path))
                else:
                    fail.append(dict(kind="OUTSIDE_REPO_MISSING", line=i, ref=path,
                                     hint="仓库外协调文件不存在"))
                continue
            # 同句内显式声明缺失 → 如实披露；显式标注「上游/仓库外」→ 外部引用
            seg = ev[max(0, m.start() - 80):m.end() + 80]
            if any(k in seg for k in ABSENCE_MARKERS):
                info.append(dict(kind="DECLARED_ABSENT", line=i, ref=path))
                continue
            if any(k in ev[max(0, m.start() - 30):m.start()] for k in EXTERNAL_MARKERS):
                info.append(dict(kind="EXTERNAL_DECLARED", line=i, ref=path))
                continue
            if "/" in path:
                if (repo / path).exists():
                    if loc:
                        nums = [int(x) for x in re.findall(r"\d+", loc)]
                        n = len((repo / path).read_text(encoding="utf-8", errors="ignore").splitlines())
                        if nums and max(nums) > n:
                            fail.append(dict(kind="LINE_OUT_OF_RANGE", line=i, ref=f"{path}:{loc}",
                                             hint=f"文件只有 {n} 行"))
                    continue
                if (repo / head).exists():        # 顶层项存在 → 这是仓库内路径，必须存在
                    fail.append(dict(kind="PATH_MISSING", line=i, ref=path,
                                     hint="仓库内相对路径不存在"))
                else:                            # 首段不是顶层项 → 视为外部/相对片段引用
                    same = sorted({*by_base.get(pathlib.Path(path).name, [])})
                    info.append(dict(kind="EXTERNAL_OR_FRAGMENT", line=i, ref=path, candidates=same))
                continue
            # 裸文件名
            cands = sorted({*by_base.get(path, [])})
            cands = [c for c in cands if c not in ("",)]
            if not cands and (repo / path).exists():
                cands = [path]
            if not cands:
                fail.append(dict(kind="PATH_MISSING", line=i, ref=path,
                                 hint="裸文件名在 tracked 文件里找不到"))
            elif len(cands) > 1:
                info.append(dict(kind="AMBIGUOUS_BASENAME", line=i, ref=path, candidates=cands))
            else:
                info.append(dict(kind="RESOLVED_BASENAME", line=i, ref=path, candidates=cands))

    # ---- ② § 章节号引用（全文） ----
    self_ids = audit_headings(audit_path)
    # 本文档标题里出现的 §号 是**外部**章节（如 §1.3 指考核 PDF 的 1.3 节）——由标题自身声明，
    # 故从标题文本里自动收集，无需维护硬编码表。
    external_ids: set[str] = set()
    for ln in lines:
        if ln.lstrip().startswith("#"):
            for m in SECTION_RE.finditer(ln):
                external_ids |= {m.group(1)}
    doc_cache: dict[str, set[str]] = {}
    for i, ln in enumerate(lines, 1):
        for m in SECTION_RE.finditer(ln):
            sec = m.group(1)
            n_secs += 1
            left = ln[:m.start()]
            clause = re.split("[" + re.escape(CLAUSE_DELIMS) + "]", left)[-1]
            sentence = re.split(r"[。|]", left)[-1][-200:]
            sec_ok_self = bool(id_variants(sec) & self_ids)
            self_trigger = bool(clause) and any(clause.endswith(t) for t in SELF_TRIGGERS)
            if self_trigger and sec_ok_self:
                info.append(dict(kind="SELF_REF", line=i, sec=sec, how="自指触发器"))
                continue
            toks = [t.group(0) for t in DOC_TOKEN_RE.finditer(sentence)]
            if toks:
                tok = toks[-1]
                rel = DOC_TOKENS[tok]
                if rel not in doc_cache:
                    target = (repo.parent / rel) if rel.startswith("_coord/") else (repo / rel)
                    doc_cache[rel] = heading_ids(target)
                ids = doc_cache[rel]
                if id_variants(sec) & ids:
                    info.append(dict(kind="CROSSDOC_REF", line=i, sec=sec, doc=rel))
                else:
                    fail.append(dict(kind="SECTION_MISSING_CROSSDOC", line=i, sec=sec, doc=rel,
                                     hint=f"{rel} 无 §{sec}（该文档编号样例："
                                          + ",".join(sorted(x for x in ids if x[:1].isdigit())[:10]) + "）"))
                continue
            if re.fullmatch(r"[一二三四五六七八九十]+", sec) or sec in external_ids:
                # 本文档自身章节用阿拉伯编号；中文 §号 与「标题里声明过的外部 §号」（如考核 PDF §1.3）
                # 一律视为外部引用（考核 PDF 不入仓库，无法机检其内部编号）
                info.append(dict(kind="PDF_SECTION_EXTERNAL", line=i, sec=sec))
                continue
            if sec_ok_self:
                info.append(dict(kind="SELF_REF", line=i, sec=sec, how="本文档章节"))
            else:
                fail.append(dict(kind="SECTION_MISSING_SELF", line=i, sec=sec,
                                 hint="本文档无该编号标题（自指），且句中无被引文档名"))

    # ---- ③ 可重算断言（t30）：计数 / 体积 / 行数 / 数据行 / grep 命中 / §8 统计求和 ----
    def _add(kind: str, ln: int, what: str, claimed, actual, unit: str = ""):
        nonlocal n_assert
        n_assert += 1
        ok = str(claimed) == str(actual)
        assert_rows.append(dict(kind=kind, line=ln, what=what, claimed=str(claimed),
                                actual=str(actual), unit=unit, ok=ok))
        if not ok:
            fail.append(dict(kind=kind + "_MISMATCH", line=ln,
                             ref=what, hint=f"声明 {claimed}{unit} / 实测 {actual}{unit}"))
        else:
            info.append(dict(kind=kind + "_OK", line=ln, ref=what))

    n_assert = 0
    assert_rows: list[dict] = []
    onnx_tracked = [f for f in tracked if f.startswith("onnx/") and f.endswith(".onnx")]
    onnx_official = [f for f in onnx_tracked if f.endswith("_in1k.onnx")]
    for i, ln in enumerate(lines, 1):
        for m in COUNT_TOTAL_RE.finditer(ln):
            _add("COUNT_ASSERTION", i, "`git ls-files onnx/` 计数", m.group(1), len(onnx_tracked), " 个")
        for m in COUNT_OFFICIAL_RE.finditer(ln):
            _add("COUNT_ASSERTION_OFFICIAL", i, "官方 ONNX 计数", m.group(1), len(onnx_official), " 个")
        for m in SIZE_RE.finditer(ln):
            rel, claimed = m.group(1), int(m.group(2).replace(",", ""))
            p = repo / rel
            if rel in SELF_REF_FILES:
                n_assert += 1
                info.append(dict(kind="SELF_REFERENCE_SKIPPED", line=i, ref=rel,
                                 how="自指脚本无法断言自身体积"))
                continue
            actual = p.stat().st_size if p.exists() else -1
            _add("SIZE_ASSERTION", i, rel, claimed, actual, " B")
        for m in LINES_RE.finditer(ln):
            rel, claimed = m.group(1), int(m.group(2).replace(",", ""))
            p = repo / rel
            if rel in SELF_REF_FILES:
                n_assert += 1
                info.append(dict(kind="SELF_REFERENCE_SKIPPED", line=i, ref=rel,
                                 how="自指脚本无法断言自身行数"))
                continue
            actual = (len(p.read_text(encoding="utf-8", errors="ignore").splitlines())
                      if p.exists() else -1)
            _add("LINES_ASSERTION", i, rel, claimed, actual, " 行")
        for m in DATAROWS_RE.finditer(ln):
            rel, claimed = m.group(1), int(m.group(2).replace(",", ""))
            p = repo / rel
            if not p.exists():
                _add("DATAROWS_ASSERTION", i, rel, claimed, -1, " 行数据")
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if rel.endswith(".csv"):
                actual = max(len(txt) - 1, 0)                      # 去表头
            else:
                actual = max(len([x for x in txt if x.strip().startswith("|")]) - 2, 0)  # 去表头+分隔
            _add("DATAROWS_ASSERTION", i, rel, claimed, actual, " 行数据")
        # grep 断言：命令与紧随其后的「**N 命中 / 个文件 / 行**」
        for m in GREP_CMD_RE.finditer(ln):
            flag = m.group(1) or m.group(5)
            pat = m.group(3) or m.group(7)
            tail = (m.group(4) or m.group(8) or "").strip()
            claim = (GREP_CLAIM_RE.search(ln[m.end():m.end() + 40])
                     or GREP_CLAIM_RE2.search(ln[m.end():m.end() + 40]))
            if not claim:
                continue
            claimed, unit = int(claim.group(1)), claim.group(2)
            args = ["git", "grep", f"-{flag}", "--", pat]
            for a_tok in re.findall(r"--\s+('[^']+'|\"[^\"]+\"|\S+)", tail):
                args += ["--", a_tok.strip("'\"")]
            if flag == "F":
                args = ["git", "grep", "-F", "--", pat] + args[5:]
            r = subprocess.run(args, cwd=repo, capture_output=True, text=True, encoding="utf-8")
            out = [x for x in r.stdout.splitlines() if x.strip()]
            if unit == "命中":
                actual = len(out)
            elif unit == "个文件":
                actual = len(out)
            else:                                                   # -cE → Σ 每文件计数
                actual = sum(int(x.rsplit(":", 1)[1]) for x in out if x.rsplit(":", 1)[-1].isdigit())
            _add("GREP_ASSERTION", i, f"git grep -{flag} {pat[:28]}", claimed, actual, f" {unit}")

    # ---- ④ §8 判定汇总：13 区块逐列求和必须等于合计行 ----
    blocks = []
    for ln in lines:
        if re.match(r"^\| §", ln) or ln.startswith("| **合计**"):
            c = [x.strip() for x in ln.strip().strip("|").split("|")]
            nums = [x.strip("*").strip() for x in c[1:6]]
            if len(nums) == 5 and all(x.isdigit() for x in nums):
                blocks.append((c[0], [int(x) for x in nums]))
    totals = [b for b in blocks if b[0].startswith("**合计**")]
    if totals:
        sums = [sum(b[1][k] for b in blocks if not b[0].startswith("**合计**")) for k in range(5)]
        n_assert += 1
        same = sums == totals[0][1]
        assert_rows.append(dict(kind="STATS_SUM", line=0, what="§8 判定汇总 13 区块逐列求和",
                                claimed=str(totals[0][1]), actual=str(sums), unit="", ok=same))
        if not same:
            fail.append(dict(kind="STATS_SUM_MISMATCH", line=0, ref="§8 判定汇总",
                             hint=f"逐列求和 {sums} ≠ 合计行 {totals[0][1]}"))
        else:
            info.append(dict(kind="STATS_SUM_OK", line=0, ref="§8 判定汇总",
                             how=f"{len(blocks)-1} 区块求和一致"))

    # ---- ⑤ 修复方案列：不得引用已删除路径 / 死指令（t34 新增）----
    # 规则：§9 行的「最小修复方案」列（5 列表格第 4 列）与任何含「修复见/修复方案」的缺口列里，
    #       出现仓库内已不存在的路径（既不在 git ls-files、也不在磁盘上）→ FAIL；
    #       仅当该引用前 160 字符内有「历史」标注（保留为历史 / 不需再执行 / t2 时点）时才放行。
    HIST_MARKERS = ("保留为历史", "不需再执行", "仅供历史", "t2 时点的修复方案",
                    "不得恢复", "已 git rm 删除", "已 `git rm` 删除", "已删除且不得恢复")
    FUTURE_MARKERS = ("落盘", "新增", "产出", "待生成", "将生成", "（新增", "若")   # 计划里的「待产出」目标
    tracked_set = set(tracked)
    base_index: dict[str, list[str]] = {}
    for f in tracked:
        base_index.setdefault(pathlib.Path(f).name, []).append(f)

    def _resolve(rel: str) -> str | None:
        """返回可定位的真实路径；裸文件名按 tracked 唯一后缀解析（与正文引用同口径）。"""
        rel = rel.strip()
        if rel in tracked_set or (repo / rel).exists():
            return rel
        if "/" not in rel:
            cands = base_index.get(rel, [])
            if len(cands) == 1:
                return cands[0]
        return None

    plan_cells = []
    for i, ln in enumerate(lines, 1):
        if not ln.startswith("|"):
            continue
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) >= 5 and re.match(r"^\*{0,2}[A-Z]{1,2}-\d+", c[0].strip("*")):
            plan_cells.append((i, c[3], c[0]))                      # §9 行：最小修复方案列
            plan_cells.append((i, c[4], c[0] + " 需改动路径列"))      # §9 行：需改动路径列（t38 起同扫）
        if len(c) >= 4 and ("修复见" in c[-2] or "修复见" in c[-1]):
            plan_cells.append((i, c[-2] if "修复见" in c[-2] else c[-1], c[0]))

    PLAN_PATH_RE = re.compile(r"`([A-Za-z0-9_./\-\u4e00-\u9fff*<>…]+?\.(?i:" + EXT_ALT + r"))`")
    PLAN_CMD_RE = re.compile(r"`((?:python|python3|sh|bash|powershell)\s+[^`]+)`")
    INPUT_FLAG_RE = re.compile(r"--(images|weights|data|cfg|source|data-list)\s+([^\s`]+)")
    for lineno, cell, key in plan_cells:
        for m in PLAN_PATH_RE.finditer(cell):
            rel = m.group(1)
            if any(ch in rel for ch in "*<>…") or rel.startswith("_coord/") or ".." in rel:
                continue
            n_assert += 1
            prev = cell[max(0, m.start() - 160):m.start()]
            hist = any(k in prev for k in HIST_MARKERS)
            future = any(k in cell[max(0, m.start() - 40):m.end() + 40] for k in FUTURE_MARKERS)
            target = _resolve(rel)
            ok_ref = target is not None
            assert_rows.append(dict(kind="PLAN_PATH", line=lineno, what=f"{key} 修复方案引用的 {rel}",
                                    claimed="存在", actual=f"存在（{target}）" if ok_ref else "不存在",
                                    unit="", ok=ok_ref or hist or future))
            if ok_ref:
                info.append(dict(kind="PLAN_PATH_OK", line=lineno, ref=rel))
            elif hist:
                info.append(dict(kind="PLAN_PATH_HISTORICAL", line=lineno, ref=rel))
            elif future:
                info.append(dict(kind="PLAN_PATH_TO_CREATE", line=lineno, ref=rel,
                                 how="计划里的待产出目标（允许尚不存在）"))
            else:
                fail.append(dict(kind="PLAN_DEAD_PATH", line=lineno, ref=rel,
                                 hint=f"{key} 的修复方案列引用了不存在/已删除的路径（且无历史/待产出标注）"))

        # 命令里的脚本路径与「输入类」参数路径同样必须存在（F-9 的形态：--images 指向已删除的清单）
        for cm in PLAN_CMD_RE.finditer(cell):
            cmd = cm.group(1)
            parts = cmd.split()
            targets = []
            if len(parts) > 1 and parts[0].startswith(("python", "python3", "sh", "bash", "powershell")):
                targets.append(parts[1])
            for fm in INPUT_FLAG_RE.finditer(cmd):
                targets.append(fm.group(2))
            for t in targets:
                t = t.strip().strip("'\"")
                if not t or t.startswith("-") or t.startswith("<") or "/" not in t and "." not in t:
                    continue
                n_assert += 1
                prev = cell[max(0, cm.start() - 160):cm.start()]
                hist = any(k in prev for k in HIST_MARKERS)
                target = _resolve(t)
                ok_ref = target is not None
                assert_rows.append(dict(kind="PLAN_CMD_PATH", line=lineno,
                                        what=f"{key} 修复方案命令里的 {t}",
                                        claimed="存在", actual=f"存在（{target}）" if ok_ref else "不存在",
                                        unit="", ok=ok_ref or hist))
                if ok_ref:
                    info.append(dict(kind="PLAN_CMD_PATH_OK", line=lineno, ref=t))
                elif hist:
                    info.append(dict(kind="PLAN_CMD_PATH_HISTORICAL", line=lineno, ref=t))
                else:
                    fail.append(dict(kind="PLAN_DEAD_PATH", line=lineno, ref=t,
                                     hint=f"{key} 的修复方案命令引用了不存在/已删除的路径（且无历史标注）"))

    # ---- ⑥ 同一文件内同类统计值须口径一致（t34 新增）----
    # 权威值 = §8 合计行的 满足/部分/缺失/挂起；文中任何「满足 N / 部分 M」都必须等于它，
    # 除非该处前 60 字符内标注为历史时点（当时 / t10 时点 / 历史 / 原）。
    if totals:
        want = (totals[0][1][1], totals[0][1][2])
        STATS_PAIR_RE = re.compile(r"满足\s*\*{0,2}(\d+)\*{0,2}\s*/\s*部分\s*\*{0,2}(\d+)\*{0,2}")
        for i, ln in enumerate(lines, 1):
            if ln.startswith("| **合计**"):
                continue
            for m in STATS_PAIR_RE.finditer(ln):
                got = (int(m.group(1)), int(m.group(2)))
                n_assert += 1
                prev = ln[max(0, m.start() - 60):m.start()]
                hist = any(k in prev for k in ("当时", "t10 时点", "历史", "原"))
                same = got == want
                assert_rows.append(dict(kind="STATS_PAIR", line=i, what="判定统计 满足/部分",
                                        claimed=f"{got[0]}/{got[1]}", actual=f"{want[0]}/{want[1]}",
                                        unit="", ok=same or hist))
                if same:
                    info.append(dict(kind="STATS_PAIR_OK", line=i, ref=f"{got[0]}/{got[1]}"))
                elif hist:
                    info.append(dict(kind="STATS_PAIR_HISTORICAL", line=i, ref=f"{got[0]}/{got[1]}"))
                else:
                    fail.append(dict(kind="STATS_INCONSISTENT", line=i, ref=f"满足 {got[0]} / 部分 {got[1]}",
                                     hint=f"与 §8 权威值 满足 {want[0]} / 部分 {want[1]} 不一致且无历史标注"))

    return dict(audit=str(audit_path), rows=n_rows, path_refs=n_paths, sec_refs=n_secs,
                assertions=n_assert, stats_blocks=max(len(blocks) - 1, 0),
                plan_cells=len(plan_cells), assertions_detail=assert_rows,
                fails=fail, infos=info,
                info_kinds={k: sum(1 for x in info if x["kind"] == k)
                            for k in sorted({x["kind"] for x in info})},
                fail_kinds={k: sum(1 for x in fail if x["kind"] == k)
                            for k in sorted({x["kind"] for x in fail})})


def summarize(res: dict) -> str:
    ok = not res["fails"]
    return (f"证据指针 {'OK' if ok else 'FAIL'}：行 {res['rows']} / 路径引用 {res['path_refs']} / "
            f"§引用 {res['sec_refs']} / 可重算断言 {res.get('assertions', 0)}；失败 {len(res['fails'])} 处")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", default=str(DEFAULT_AUDIT))
    ap.add_argument("--json", dest="json_out")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--show-info", action="store_true", help="打印 INFO 明细")
    a = ap.parse_args(argv)

    res = scan(a.audit)
    if a.json_out:
        pathlib.Path(a.json_out).write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    if a.quiet:
        print(summarize(res))
    else:
        print(f"审计文档: {res['audit']}")
        print(f" 数据行(满足/部分) = {res['rows']}；路径引用 = {res['path_refs']}；§引用 = {res['sec_refs']}")
        print(f" INFO 分类 = {res['info_kinds']}")
        print(f" FAIL 分类 = {res['fail_kinds']}")
        for k in res["fail_kinds"]:
            print(f"\n--- FAIL {k} ---")
            for f in res["fails"]:
                if f["kind"] == k:
                    extra = f" [{f.get('doc')} §{f.get('sec')}]" if f.get("sec") else f" `{f.get('ref')}`"
                    print(f"  L{f['line']}: {extra}  {f.get('hint','')}")
        if a.show_info and res["infos"]:
            print("\n--- INFO ---")
            for x in res["infos"]:
                print(f"  L{x['line']}: {x['kind']} {x.get('ref','')} {x.get('candidates','')}")
        print()
        print(summarize(res))
    return 0 if not res["fails"] else 1


if __name__ == "__main__":
    sys.exit(main())
