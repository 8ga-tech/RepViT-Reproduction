# -*- coding: utf-8 -*-
"""扫描全仓 logits 误差引用，并按实验口径分级。

用法（仓库根目录）::

    python tools/audit_logits_references.py            # 刷新 outputs/verification/logits_reference_inventory.csv
    python tools/audit_logits_references.py --table    # 额外打印「实验桶 → 权威产物」Markdown 表

分级原则
--------
**一个数值属于哪个实验桶，由「该数值出现在哪个落盘产物里」决定，而不是由散文字面决定。**

脚本运行时先把各权威产物的数值读成「规范值集合」；扫描到的每个浮点 token，只有当它等于
某个规范值（按该 token 自身书写的有效位数四舍五入后仍相等）时才归入对应实验桶。这样
``6.199e-06``（B1 的展示写法）与 ``6.53e-06``（B8 孤值）不会被混为一谈，也不需要把口径
写死在正则里。匹配不上规范值的浮点 token 退回上下文关键词判定，或标为 ``noise_float``
（学习率、概率、指标列等只是长得像误差值的数字）。

覆盖范围（仅 Git 跟踪文件）
--------------------------
* 文本/代码/JSON/CSV/日志/YAML/SVG：按行，location = 行号。
* PPTX：``ppt/slides/slideN.xml`` 逐段（location 记段号）、``ppt/notesSlides/notesSlideN.xml``
  逐段（location 记所属幻灯片号）、``ppt/charts/chartN.xml`` 的数值缓存（``c:v``）、
  图表内嵌 ``ppt/embeddings/*.xlsx`` 的数据缓存。
* DOCX：``word/document.xml`` 与 header/footer 逐段。
* PDF：用 PyMuPDF 逐页提取文本，location = ``page N``。

输出列
------
``file / location / reference / match / kind / bucket / authority / reproducible / flag``

* ``kind``：``error_value``（数值引用）/ ``statement``（口径陈述）/ ``noise_float``（只是长得像误差值的数字）。
* ``bucket``：实验桶，B1–B10 见 ``BUCKETS``；``B11_statement`` 为口径/方法陈述；``NOT_AN_ERROR_REF`` 为噪音过滤桶。
* ``flag``：``ORPHAN``（孤值）/ ``MIXED``（同一行混引多个桶）/ ``OVERCLAIM``（超出证据的断言）/
  ``ABS_PATH``（出现本机绝对路径）/ ``AMBIGUOUS``（token 同时命中多个桶）/ ``SELF_TOOL``（本脚本自身的孤值登记表）。

不参与扫描：二进制权重/模型/图片（``.pt`` / ``.onnx`` / ``.npy`` / 图片），以及本清单自身。
``tools/audit_logits_references.py`` 会照常扫描，但带 ``SELF_TOOL`` 标记——那些行是审计脚本自己的
规范值登记表，不是外部引用点，统计时应当剔除。
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT_REL = 'outputs/verification/logits_reference_inventory.csv'
OUT = ROOT / OUT_REL
SELF_TOOL_REL = 'tools/audit_logits_references.py'

TEXT_SUFFIXES = {'.md', '.json', '.jsonl', '.csv', '.log', '.txt', '.py', '.sh', '.svg', '.yaml', '.yml'}
FLOAT_TOKEN = re.compile(r'\d+\.\d+[eE]-0\d')
SCAN = re.compile(r'logits|重参数化|一致(?:性|率)|误差|rel_err|[1-9]\.\d+e-0[1-9]|PyTorch.{0,8}ONNX', re.I)
# 判「这个数字是误差值」而不是「学习率/概率」的上下文关键词
ERROR_CTX = re.compile(r'logits|误差|abs_err|abs_logits|rel_err|Δ|一致性', re.I)
OVERCLAIM = re.compile(r'(?:六|6\s*个)\s*ONNX|六个模型|全部六|完全等价|九类(?:典型)?问题')
ABS_PATH = re.compile(r'[A-Za-z]:[\\/]{1,2}Users', re.I)

# --------------------------------------------------------------------------- #
# 实验桶定义：每个桶对应唯一的权威落盘产物
# --------------------------------------------------------------------------- #
BUCKETS = {
    'B1_onnx_pet37_n12': dict(
        title='PyTorch↔ONNX｜Pet-37 baseline｜n=12 张真实图片',
        authority='outputs/metrics/consistency_repvit_m0_9_pet37.json',
        display='max 6.199e-06 / mean 1.501e-06；Top-1 与 Top-5 集合一致率 1.000；verdict PASS',
        repro='yes',
        cmd='python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json',
    ),
    'B2_onnx_m0_9_in1k_n12': dict(
        title='PyTorch↔ONNX｜M0.9 官方 ImageNet 子集｜n=12 张真实图片',
        authority='outputs/metrics/consistency_repvit_m0_9_in1k.json',
        display='max 1.717e-05 / mean 2.360e-06',
        repro='yes',
        cmd='python deploy/compare_torch_onnx.py --model repvit_m0_9_in1k --images datasets/lists/imagenet_val_subset.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_in1k_n12.json',
    ),
    'B3_onnx_m1_0_in1k_n12': dict(
        title='PyTorch↔ONNX｜M1.0 官方 ImageNet 子集｜n=12 张真实图片',
        authority='outputs/metrics/consistency_repvit_m1_0_in1k.json',
        display='max 1.812e-05 / mean 2.464e-06',
        repro='yes',
        cmd='python deploy/compare_torch_onnx.py --model repvit_m1_0_in1k --images datasets/lists/imagenet_val_subset.txt --limit 12 --out outputs/verification/consistency_repvit_m1_0_in1k_n12.json',
    ),
    'B4_reparam_pet37_32rand': dict(
        title='结构重参数化｜Pet-37 baseline｜32 个固定随机输入 seed=20240912',
        authority='outputs/reparam/repvit_m0_9_pet37_reparam_report.json',
        display='max 7.093e-06 / mean 2.031e-06；Top-1 32/32；BN 107→0',
        repro='yes',
        cmd='python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37',
    ),
    'B5_official_fuse_probe': dict(
        title='结构重参数化｜五个官方 ImageNet 型号｜评价流程内单输入融合探针',
        authority='outputs/pretrained_eval/*/metrics.json（fuse_max_abs_logits_diff）',
        display='M0.9 3.073e-05；M1.0 3.290e-05；M1.1 2.217e-05；M1.5 2.587e-05；M2.3 5.925e-05',
        repro='yes',
        cmd='python tools/eval_pretrained.py --cfg configs/pretrained_eval.yaml',
    ),
    'B6_export_probe': dict(
        title='导出检查｜单个随机输入前后向探针（导出脚本未固定 seed）',
        authority='outputs/benchmarks/export_*.json',
        display='按文件记录值；不充当 n=12 真实图片一致性结果',
        repro='partial',
        cmd='python deploy/export_onnx.py --model repvit_m0_9_pet37',
    ),
    'B7_reparam_legacy_invalid': dict(
        title='结构重参数化｜历史/无效口径记录（不作为当前结论）',
        authority='outputs/reparam/repvit_m0_9_reparam_report.json（C=1000 官方权重装入 timm 结构，n_missing=605）；'
                  'outputs/advanced/repvit_m0_9_pet37_reparam_report.json（seed=0、2 个随机输入的历史结构探针）',
        display='4.991888999938965e-07 / 2.9802322387695312e-06（仅作历史记录）',
        repro='no',
        cmd='（不提供复跑命令：产物对应无效权重或已废弃探针）',
    ),
    'B8_legacy_orphan': dict(
        title='孤值｜仓库内找不到配套产物，也没有任何提交能佐证其样本数',
        authority='（无）',
        display='5.25e-06 / 1.41e-06、5.245e-06 / 1.414e-06、6.53e-06、6.527e-06、2.265e-06、4.108e-07',
        repro='no',
        cmd='（无可复跑产物：当前权重 n=8 / n=12 均得不到这些数值）',
    ),
    'B9_route_probe': dict(
        title='路线等价探针｜两条权重路线的特征/主头/蒸馏平均 logits 比较',
        authority='outputs/metrics/probe_route.json',
        display='max=0（同进程同权重路线对比，不是 ONNX 或融合实验）',
        repro='yes',
        cmd='python tools/probe_pretrained_route.py --official-ckpt checkpoints/pretrained/repvit_m0_9_distill_300e.pth --out outputs/metrics/probe_route.json',
    ),
    'B10_smoke_early': dict(
        title='早期冒烟与结构统计｜smoke_phase0 与结构 txt',
        authority='outputs/metrics/smoke_phase0.json',
        display='按文件记录值',
        repro='partial',
        cmd='python tools/smoke_phase0.py',
    ),
    'B11_statement': dict(
        title='口径/方法陈述｜提到 logits 误差但没有可归因数值',
        authority='（方法或口径说明，不指向具体数值）',
        display='—',
        repro='-',
        cmd='—',
    ),
    'NOT_AN_ERROR_REF': dict(
        title='非误差引用｜只是长得像误差值的数字（学习率、概率、指标列）或纯代码标识符',
        authority='—',
        display='—',
        repro='-',
        cmd='—',
    ),
}

# 孤值：逐字面量登记，避免与相邻桶的规范值相互污染
ORPHAN_LITERALS = ['5.25e-06', '5.245e-06', '1.41e-06', '1.414e-06',
                   '6.53e-06', '6.527e-06', '2.265e-06', '4.108e-07']

PATH_RULES = [
    ('outputs/metrics/consistency_repvit_m0_9_pet37.json', 'B1_onnx_pet37_n12'),
    ('outputs/verification/consistency_repvit_m0_9_pet37_n12.json', 'B1_onnx_pet37_n12'),
    ('outputs/metrics/consistency_repvit_m0_9_in1k.json', 'B2_onnx_m0_9_in1k_n12'),
    ('outputs/metrics/consistency_repvit_m1_0_in1k.json', 'B3_onnx_m1_0_in1k_n12'),
    ('outputs/reparam/repvit_m0_9_pet37_reparam_report.json', 'B4_reparam_pet37_32rand'),
    ('outputs/reparam/logits_diff.json', 'B4_reparam_pet37_32rand'),
    ('outputs/verification/reparam_pet37/', 'B4_reparam_pet37_32rand'),
    ('outputs/advanced/repvit_m0_9_pet37_reparam_report.json', 'B7_reparam_legacy_invalid'),
    ('outputs/reparam/repvit_m0_9_reparam_report.json', 'B7_reparam_legacy_invalid'),
    ('outputs/pretrained_eval/', 'B5_official_fuse_probe'),
    ('outputs/benchmarks/export_', 'B6_export_probe'),
    ('deploy/export_onnx.py', 'B6_export_probe'),
    ('outputs/metrics/probe_route.json', 'B9_route_probe'),
    ('tools/probe_pretrained_route.py', 'B9_route_probe'),
    ('outputs/metrics/smoke_phase0.json', 'B10_smoke_early'),
    ('tools/smoke_phase0.py', 'B10_smoke_early'),
]


# --------------------------------------------------------------------------- #
# 数值匹配：token 与规范值是否一致（按 token 自身书写的有效位数四舍五入）
# --------------------------------------------------------------------------- #
def _sig_digits(token: str) -> int:
    mant = re.split(r'[eE]', token)[0]
    return max(len(mant.replace('.', '').lstrip('0')), 1)


def _exp10(value: float) -> int:
    return math.floor(math.log10(abs(value)))


def token_matches(token: str, value: float) -> bool:
    """token 在它自己写出的精度上，是否就是 value。"""
    t = float(token)
    if t == 0:
        return False
    half = 0.5 * (10.0 ** (_exp10(t) - (_sig_digits(token) - 1)))
    return abs(value - t) <= half * (1 + 1e-9)


def collect_canonical() -> tuple[dict[str, set[float]], list[str]]:
    """从权威落盘产物读出「实验桶 → 规范值集合」。产物缺失时跳过并给出告警。"""
    import json

    values: dict[str, set[float]] = {}
    warnings: list[str] = []

    def add(bucket: str, *vals):
        for v in vals:
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values.setdefault(bucket, set()).add(float(v))

    def load(rel: str):
        p = ROOT / rel
        if not p.is_file():
            warnings.append(f'缺少权威产物：{rel}（该桶只按路径规则分级）')
            return None
        try:
            return json.loads(p.read_text(encoding='utf-8'))
        except Exception as exc:                                # pragma: no cover
            warnings.append(f'无法解析 {rel}：{exc}')
            return None

    for rel, bucket, keys in [
        ('outputs/metrics/consistency_repvit_m0_9_pet37.json', 'B1_onnx_pet37_n12',
         ('max_abs_logits', 'mean_abs_logits', 'top1_agree_rate', 'top5_set_agree_rate')),
        ('outputs/metrics/consistency_repvit_m0_9_in1k.json', 'B2_onnx_m0_9_in1k_n12',
         ('max_abs_logits', 'mean_abs_logits', 'top1_agree_rate', 'top5_set_agree_rate')),
        ('outputs/metrics/consistency_repvit_m1_0_in1k.json', 'B3_onnx_m1_0_in1k_n12',
         ('max_abs_logits', 'mean_abs_logits', 'top1_agree_rate', 'top5_set_agree_rate')),
        ('outputs/reparam/repvit_m0_9_pet37_reparam_report.json', 'B4_reparam_pet37_32rand',
         ('max_abs_err', 'mean_abs_err')),
        ('outputs/reparam/repvit_m0_9_reparam_report.json', 'B7_reparam_legacy_invalid',
         ('max_abs_err', 'mean_abs_err')),
        ('outputs/advanced/repvit_m0_9_pet37_reparam_report.json', 'B7_reparam_legacy_invalid',
         ('max_abs_err', 'mean_abs_err')),
    ]:
        data = load(rel)
        if isinstance(data, dict):
            add(bucket, *[data.get(k) for k in keys])
            diff = data.get('diff') if isinstance(data.get('diff'), dict) else {}
            add(bucket, diff.get('max_abs_err'), diff.get('max_rel_err'))

    for p in sorted((ROOT / 'outputs/pretrained_eval').glob('*/metrics.json')):
        data = load(p.relative_to(ROOT).as_posix())
        if isinstance(data, dict):
            add('B5_official_fuse_probe', data.get('fuse_max_abs_logits_diff'))

    for p in sorted((ROOT / 'outputs/benchmarks').glob('export_*.json')):
        data = load(p.relative_to(ROOT).as_posix())
        if isinstance(data, dict):
            for k, v in data.items():
                if 'logits' in k.lower() or 'diff' in k.lower():
                    add('B6_export_probe', v)

    for lit in ORPHAN_LITERALS:                                  # 孤值：按字面量登记
        values.setdefault('B8_legacy_orphan', set()).add(float(lit))

    return values, warnings


def match_bucket(token: str, canonical: dict[str, set[float]]) -> tuple[str | None, bool]:
    """返回 (桶, 是否有歧义)。歧义表示同一 token 同时命中多个桶。"""
    hits = {b for b, vals in canonical.items() if any(token_matches(token, v) for v in vals)}
    if not hits:
        return None, False
    if len(hits) > 1:
        return sorted(hits)[0], True
    return hits.pop(), False


def accept_value(rel: str, text: str, token: str) -> bool:
    """纯数值日志里学习率/损失也写成 e-0X，需要更严的接受条件。

    规则：行内有误差上下文（logits/误差/abs_err/Δ…）就接受；否则只有在
    ``outputs/logs``、``outputs/metrics`` 的逐行数值文件里才要求 token 至少 4 位有效数字
    （学习率一般只写 3 位，例如 ``lr_bb=1.81e-05``，避免误判成 M1.0 的 max|Δ|）。
    """
    if ERROR_CTX.search(text):
        return True
    low = rel.lower()
    numeric_log = ('outputs/logs/' in low or 'outputs/metrics/' in low) and \
        rel.endswith(('.log', '.jsonl', '.csv'))
    return _sig_digits(token) >= 4 if numeric_log else True


# --------------------------------------------------------------------------- #
# 采集
# --------------------------------------------------------------------------- #
def _pptx_index(zf: zipfile.ZipFile) -> tuple[dict[int, int], dict[int, int]]:
    """返回 (notesSlide 号 → slide 号, chart 号 → slide 号)。"""
    notes, charts = {}, {}
    for name in zf.namelist():
        m = re.fullmatch(r'ppt/slides/_rels/slide(\d+)\.xml\.rels', name)
        if m:
            for target in re.findall(r'Target="([^"]+)"', zf.read(name).decode('utf-8', 'replace')):
                c = re.search(r'charts/chart(\d+)\.xml', target)
                if c:
                    charts[int(c.group(1))] = int(m.group(1))
        m = re.fullmatch(r'ppt/notesSlides/_rels/notesSlide(\d+)\.xml\.rels', name)
        if m:
            for target in re.findall(r'Target="([^"]+)"', zf.read(name).decode('utf-8', 'replace')):
                s = re.search(r'slides/slide(\d+)\.xml', target)
                if s:
                    notes[int(m.group(1))] = int(s.group(1))
    return notes, charts


def _paragraphs(root, tag_suffix: str) -> list[str]:
    out = []
    for para in root.iter():
        if para.tag.endswith('}' + tag_suffix):
            out.append(' '.join(t.text or '' for t in para.iter() if t.tag.endswith('}t')))
    return out


def iter_units(rel: str):
    """产出 (location, text) 二元组。"""
    path = ROOT / rel
    suffix = path.suffix.lower()
    if suffix in ('.pptx', '.docx'):
        with zipfile.ZipFile(path) as zf:
            if suffix == '.pptx':
                notes_map, chart_map = _pptx_index(zf)
                for name in zf.namelist():
                    m = re.fullmatch(r'ppt/slides/slide(\d+)\.xml', name)
                    if m:
                        n = int(m.group(1))
                        for i, text in enumerate(_paragraphs(ET.fromstring(zf.read(name)), 'p'), 1):
                            yield f'slide {n} para {i} ({name})', text
                        continue
                    m = re.fullmatch(r'ppt/notesSlides/notesSlide(\d+)\.xml', name)
                    if m:
                        n = int(m.group(1))
                        host = notes_map.get(n)
                        loc = f'notes {n}' + (f' for slide {host}' if host else '')
                        for i, text in enumerate(_paragraphs(ET.fromstring(zf.read(name)), 'p'), 1):
                            yield f'{loc} para {i} ({name})', text
                        continue
                    m = re.fullmatch(r'ppt/charts/chart(\d+)\.xml', name)
                    if m:
                        n = int(m.group(1))
                        host = chart_map.get(n)
                        loc = f'chart {n}' + (f' on slide {host}' if host else '')
                        vals = [e.text for e in ET.fromstring(zf.read(name)).iter()
                                if e.tag.endswith('}v') and e.text]
                        yield f'{loc} data cache ({name})', ' '.join(vals)
                        continue
                    m = re.fullmatch(r'ppt/embeddings/(.+\.xlsx)', name)
                    if m:
                        inner = zipfile.ZipFile(io.BytesIO(zf.read(name)))
                        vals = []
                        for part in inner.namelist():
                            if part.endswith('.xml') and 'sheet' in part:
                                vals += re.findall(r'<v>([^<]+)</v>', inner.read(part).decode('utf-8', 'replace'))
                        yield f'embedded chart data ({name})', ' '.join(vals)
                        continue
            else:  # .docx
                for name in zf.namelist():
                    if re.fullmatch(r'word/(?:document|header\d*|footer\d*|footnotes|endnotes)\.xml', name):
                        for i, text in enumerate(_paragraphs(ET.fromstring(zf.read(name)), 'p'), 1):
                            yield f'{name} para {i}', text
    elif suffix == '.pdf':
        import fitz
        with fitz.open(path) as doc:
            for i, page in enumerate(doc):
                yield f'page {i + 1}', page.get_text().replace('\n', ' ')
    elif suffix in TEXT_SUFFIXES:
        for i, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            yield str(i), line


def context_bucket(rel: str, location: str, text: str) -> str:
    for prefix, bucket in PATH_RULES:
        if rel == prefix or rel.startswith(prefix):
            return bucket
    low = (rel + ' ' + text).lower()
    onnx = 'onnx' in low
    reparam = '重参数化' in text or 'reparam' in low or '融合' in text or 'fuse' in low
    pet = 'pet37' in low or 'pet-37' in low or 'pet_test' in low
    in1k = 'in1k' in low or 'imagenet' in low
    if reparam and onnx and not ERROR_CTX.search(text):
        # 「… -> 重参数化 -> ONNX -> 基准」这类流程性描述，不指向任何单个实验桶
        return 'B11_statement'
    if onnx:
        if pet:
            return 'B1_onnx_pet37_n12'
        if 'm1_0' in low or 'm1.0' in low:
            return 'B3_onnx_m1_0_in1k_n12'
        if 'm0_9' in low or 'm0.9' in low or in1k:
            return 'B2_onnx_m0_9_in1k_n12'
        return 'B1_onnx_pet37_n12'
    if reparam:
        if '官方' in text or 'official' in low or 'c=1000' in low or in1k:
            return 'B5_official_fuse_probe'
        return 'B4_reparam_pet37_32rand'
    if 'probe_route' in low or '路线' in text:
        return 'B9_route_probe'
    return 'B11_statement'


def classify(rel: str, location: str, text: str, match: str,
             canonical: dict[str, set[float]]) -> tuple[str, str, bool]:
    """返回 (kind, bucket, ambiguous)。"""
    if FLOAT_TOKEN.fullmatch(match):
        bucket, ambiguous = match_bucket(match, canonical)
        if bucket and accept_value(rel, text, match):
            return 'error_value', bucket, ambiguous
        if ERROR_CTX.search(text) and not bucket:
            return 'error_value', context_bucket(rel, location, text), False
        return 'noise_float', 'NOT_AN_ERROR_REF', False
    if ERROR_CTX.search(text) or '重参数化' in text or '一致' in text:
        return 'statement', context_bucket(rel, location, text), False
    return 'noise_float', 'NOT_AN_ERROR_REF', False


def flags_for(rel: str, text: str, buckets: list[str]) -> str:
    out = []
    if rel == SELF_TOOL_REL:
        out.append('SELF_TOOL')
    real = {b for b in buckets if b not in ('NOT_AN_ERROR_REF', 'B11_statement')}
    if len(real) > 1:
        out.append('MIXED')
    if 'B8_legacy_orphan' in buckets:
        out.append('ORPHAN')
    if OVERCLAIM.search(text):
        out.append('OVERCLAIM')
    if ABS_PATH.search(text):
        out.append('ABS_PATH')
    return '+'.join(out) if out else '-'


def collect(canonical: dict[str, set[float]]) -> list[list[str]]:
    rows: list[list[str]] = []
    listed = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode('utf-8').split('\0')
    for rel in listed:
        if not rel or rel == OUT_REL or not (ROOT / rel).is_file():
            continue
        for location, text in iter_units(rel):
            if not text or not SCAN.search(text):
                continue
            line_buckets = []
            for match in SCAN.finditer(text):
                kind, bucket, ambiguous = classify(rel, location, text, match.group(0), canonical)
                line_buckets.append(bucket)
                excerpt = text[max(0, match.start() - 70):match.end() + 130].strip()
                rows.append([rel, location, excerpt, match.group(0), kind, bucket,
                             BUCKETS[bucket]['authority'], BUCKETS[bucket]['repro'],
                             ('AMBIGUOUS' if ambiguous else '')])
            if not line_buckets:
                continue
            flag = flags_for(rel, text, line_buckets)
            if flag != '-':
                for row in rows[-len(line_buckets):]:
                    row[8] = '+'.join(x for x in (row[8], flag) if x) or '-'
    rows.sort(key=lambda r: (r[0], _loc_sort_key(r[1]), r[3]))
    return rows


def _loc_sort_key(location: str):
    m = re.search(r'\d+', location)
    base = location[:m.start()] if m else location
    return (base, int(m.group(0)) if m else 0, location)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                            # pragma: no cover
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', action='store_true', help='额外打印实验桶 Markdown 表')
    ap.add_argument('--stdout', action='store_true', help='只打印摘要，不写 CSV')
    args = ap.parse_args()

    canonical, warnings = collect_canonical()
    for w in warnings:
        print(f'[warn] {w}', file=sys.stderr)
    rows = collect(canonical)

    if not args.stdout:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        with OUT.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['file', 'location', 'reference', 'match',
                             'kind', 'bucket', 'authority', 'reproducible', 'flag'])
            writer.writerows(rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row[5]] = counts.get(row[5], 0) + 1
    print(f'{len(rows)} references in {len({r[0] for r in rows})} files -> '
          f'{OUT_REL if not args.stdout else "(stdout only)"}')
    for bucket in BUCKETS:
        if counts.get(bucket):
            print(f'  {counts[bucket]:5d}  {bucket:26s} {BUCKETS[bucket]["authority"]}')
    flagged = sum(1 for r in rows if r[8] and r[8] != '-')
    print(f'  标记行：{flagged}；歧义行：{sum(1 for r in rows if "AMBIGUOUS" in r[8])}')

    if args.table:
        print('\n| 实验桶 | 权威落盘产物 | 应统一引用的展示值 | 复跑命令 |')
        print('|---|---|---|---|')
        for bucket in BUCKETS:
            if bucket in ('B11_statement', 'NOT_AN_ERROR_REF'):
                continue
            info = BUCKETS[bucket]
            print(f'| `{bucket}`<br>{info["title"]} | `{info["authority"]}` | {info["display"]} | '
                  f'`{info["cmd"]}` |')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
