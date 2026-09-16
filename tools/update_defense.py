"""重建答辩 PPT 的数据页与可视化页（不跑推理，只读落盘产物）。

唯一生成入口：``python tools/update_defense.py``

版本要点
--------
1. **15 页硬约束**：原 P03（M0.9 结构，试题建议环节 2）与原 P04（为什么不是标准 ViT，
   环节 3）合并为一页，腾出的位置新增「预测结果」页，承载试题第 10 页的三条可视化要求
   （≥8 张测试集预测 / ≥4 张 Baseline 与优化模型同图对比 / ≥5 张训练集以外的实际图片）。
   合并事实写进合并页备注，并逐条列出试题第 16–17 页 15 个建议环节的落点。
2. **数字全部现填**：图表与表格从 ``outputs/logs/*.csv``、``outputs/metrics/*.json``、
   ``outputs/benchmarks/summary.csv``、``outputs/predictions/*.csv|json``、
   ``outputs/confusion_matrix/*.csv|json`` 读取，脚本里不写死实验数值。
3. **口径以 ``report/LOGITS_AUDIT_FINDINGS.md`` 第 1 节为准**：
   结构重参数化 = B4（32 个固定随机输入，``outputs/reparam/repvit_m0_9_pet37_reparam_report.json``）；
   PyTorch↔ONNX = B1（n=12 真实图片，``outputs/metrics/consistency_repvit_m0_9_pet37.json``）。
   两者在页面上分开表述、各自带限定语；``5.25e-06`` 只能作为「未找到配套产物的旧引用」出现。
   展示值统一 4 位有效数字（7.093e-06 / 2.031e-06 / 6.199e-06 / 1.501e-06）。
4. **措辞**：正文只写「怎么做 / 看到了什么 / 还不能说明什么」，去掉超证据断言
   （见 ``REPLACEMENTS``），替换对全部 15 页正文生效（含新增页）。
5. 正文与备注同步写进 ``report/PPT_CONTENT.md``，脚本可重复运行（幂等）。

边界：本脚本只改 ``report/答辩PPT_RepViT.pptx`` 与 ``report/PPT_CONTENT.md``；
PDF 由 ``python tools/export_defense_pdf.py`` 用 PowerPoint COM 重新导出。
"""
import csv
import json
import re
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.shapes import MSO_SHAPE, MSO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PPT = ROOT / "report/答辩PPT_RepViT.pptx"
EMU = 914400
BG, WHITE, MUTED, CYAN, GREEN, ORANGE = "101B30", "E6EDF7", "B6C5DC", "55C5EF", "48D9AE", "F2B560"
R_NS = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"

# 页码/标题锚点：按标题找页，避免合并或插页后下标串位（脚本可重复运行）。
STRUCT_TITLE = "M0.9 是纯卷积流水线：名字里有 ViT，结构里没有注意力算子"
STRUCT_MARK = "个 RepViT Block 的纯卷积流水线"        # 原 P03 标题
NOT_VIT_MARK = "名字里有 ViT，但结构里没有一个注意力算子"  # 原 P04 标题
PRED_TITLE = "预测结果：8 张测试集 + 4 组同图对比 + 5 张实拍"
PRED_MARK = "预测结果：8 张测试集"
CURVES_MARK = "训练曲线：两组都收敛"
CM_MARK = "错在哪里"
CAM_MARK = "Grad-CAM：同时看判对和判错的图片"
REPARAM_MARK = "重参数化：合并分支后"
ONNX_MARK = "PyTorch ↔ ONNX"
BENCH_MARK = "部署怎么选"

# 结构重参数化（B4）与 PyTorch↔ONNX（B1）的权威落盘产物，见 LOGITS_AUDIT_FINDINGS 第 1 节。
B4_REPORT = "outputs/reparam/repvit_m0_9_pet37_reparam_report.json"
B1_REPORT = "outputs/metrics/consistency_repvit_m0_9_pet37.json"


# --------------------------------------------------------------------------- IO
def data(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def raw_rows(path):
    with (ROOT / path).open(encoding="utf-8-sig") as f:
        return list(csv.reader(f))


def distribution_stats():
    """distribution_compare_summary.csv 表头重复（width/height… 各 3 列），按首列名取 mean 列。"""
    rd = raw_rows("outputs/predictions/distribution_compare_summary.csv")
    head, first = rd[0], {}
    for i, name in enumerate(head):
        if name:
            first.setdefault(name, i)
    out = {}
    for row in rd[1:]:
        if row and row[0] in ("dataset", "external"):
            out[row[0]] = {k: float(row[i]) for k, i in first.items()}
    return out


# ---------------------------------------------------------------------- 基础图元
def text(s, value, x, y, w, h, size=17, color=WHITE, bold=False, gap=6):
    sh = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = sh.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(value.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        p.font.name = "Microsoft YaHei"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor.from_string(color)
        p.space_after = Pt(gap)
    return sh


def clear_slide(s, title, subtitle, number, source, notes="", title_size=29):
    for sh in list(s.shapes):
        s.shapes._spTree.remove(sh._element)
    # 只删形状不会断关系：被删掉的图片 / 图表（图表还带一个嵌入 xlsx）会变成孤立部件，
    # 重复运行会让文件越来越大。这里保留版式与备注关系，其余一律断开。
    for rel in list(s.part.rels.values()):
        if rel.reltype not in (RT.SLIDE_LAYOUT, RT.NOTES_SLIDE):
            s.part.drop_rel(rel.rId)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor.from_string(BG)
    text(s, title, .6, .35, 12.1, .6, title_size, bold=True)
    text(s, subtitle, .6, 1.03, 12.1, .52, 14, MUTED)
    text(s, source, .6, 7.06, 11.85, .25, 9, MUTED)
    text(s, str(number).zfill(2), 12.2, 7.03, .5, .3, 12, MUTED)
    s.notes_slide.notes_text_frame.text = notes or source


def picture(s, path, x, y, w, h):
    path = ROOT / path
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    ww, hh = iw * scale, ih * scale
    s.shapes.add_picture(str(path), Inches(x + (w - ww) / 2), Inches(y + (h - hh) / 2),
                         width=Inches(ww), height=Inches(hh))


def chart(s, title, labels, series, x, y, w, h, kind=XL_CHART_TYPE.LINE, lo=None, hi=None, fmt="0.0"):
    text(s, title, x, y, w, .3, 15, CYAN, True)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y + .34), Inches(w), Inches(h - .34))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
    bg.line.fill.background()
    cd = CategoryChartData()
    cd.categories = labels
    for name, values in series:
        cd.add_series(name, values)
    ch = s.shapes.add_chart(kind, Inches(x), Inches(y + .34), Inches(w), Inches(h - .34), cd).chart
    ch.chart_style = 10
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.include_in_layout = False
    ch.legend.font.size = Pt(10)
    ch.font.name = "Arial"
    ch.font.size = Pt(10)
    ch.font.color.rgb = RGBColor(30, 45, 65)
    ch.value_axis.tick_labels.number_format = fmt
    ch.value_axis.tick_labels.font.size = Pt(10)
    ch.category_axis.tick_labels.font.size = Pt(9)
    if lo is not None:
        ch.value_axis.minimum_scale = lo
    if hi is not None:
        ch.value_axis.maximum_scale = hi
    if len(labels) > 10:
        skip = OxmlElement("c:tickLblSkip")
        skip.set("val", "10")
        ch.category_axis._element.append(skip)
    for i, se in enumerate(ch.series):
        color = ["218EBC", "EB9951"][i % 2]
        se.format.line.color.rgb = RGBColor.from_string(color)
        se.format.line.width = Pt(1.6)
        if kind != XL_CHART_TYPE.LINE:
            se.format.fill.solid()
            se.format.fill.fore_color.rgb = RGBColor.from_string(color)
    return ch


def table(s, cells, x, y, w, h, sizes=None, font=15):
    t = s.shapes.add_table(len(cells), len(cells[0]), Inches(x), Inches(y), Inches(w), Inches(h)).table
    if sizes:
        for col, size in zip(t.columns, sizes):
            col.width = Inches(size)
    for i, row in enumerate(cells):
        for j, val in enumerate(row):
            c = t.cell(i, j)
            c.text = str(val)
            c.fill.solid()
            c.fill.fore_color.rgb = RGBColor.from_string("233A56" if i == 0 else "182941")
            c.margin_left = c.margin_right = Inches(.12)
            for p in c.text_frame.paragraphs:
                p.font.name = "Microsoft YaHei"
                p.font.size = Pt(font)
                p.font.bold = i == 0
                p.font.color.rgb = RGBColor.from_string(WHITE)
    return t


# ------------------------------------------------------------------- 幻灯片工具
def iter_shapes(shape):
    yield shape
    if getattr(shape, "shapes", None) is not None:
        for child in shape.shapes:
            yield from iter_shapes(child)


def slide_strings(slide):
    out = []
    for sh in slide.shapes:
        for sub in iter_shapes(sh):
            if getattr(sub, "has_text_frame", False) and sub.text.strip():
                out.append(sub.text)
            if getattr(sub, "has_table", False):
                for row in sub.table.rows:
                    out.append(" ".join(c.text for c in row.cells))
    return out


def slide_text(slide):
    return "\n".join(slide_strings(slide))


def find_slide(prs, marker, required=True):
    for s in prs.slides:
        if marker in slide_text(s):
            return s
    if required:
        raise RuntimeError(f"PPT 里找不到标题含「{marker}」的页；先确认幻灯片未被外部改动")
    return None


def find_slide_pos(prs, marker, required=True):
    for i, s in enumerate(prs.slides):
        if marker in slide_text(s):
            return i, s
    if required:
        raise RuntimeError(f"PPT 里找不到标题含「{marker}」的页；先确认幻灯片未被外部改动")
    return None, None


def drop_slide(prs, slide):
    for node in list(prs.slides._sldIdLst):
        if prs.part.related_part(node.get(R_NS)) is slide.part:
            prs.part.drop_rel(node.get(R_NS))
            prs.slides._sldIdLst.remove(node)
            return
    raise RuntimeError("要删除的幻灯片不在 sldIdLst 里")


def move_slide(prs, src, dst):
    lst = prs.slides._sldIdLst
    nodes = list(lst)
    lst.remove(nodes[src])
    lst.insert(dst, nodes[src])


def trim_tail(prs, keep=15):
    """历史包袱：早期版本会在末尾追加临时页；这里保证刻板上限 15 页。"""
    while len(prs.slides) > keep:
        node = prs.slides._sldIdLst[-1]
        prs.part.drop_rel(node.get(R_NS))
        prs.slides._sldIdLst.remove(node)


def set_text(tf, value):
    """只改文字、保留第一个 run 的字体格式。"""
    p = tf.paragraphs[0]
    if p.runs:
        p.runs[0].text = value
        for r in p.runs[1:]:
            r.text = ""
    else:
        p.text = value


def ensure_footnote(slide, value, x, y, w, h, size, color=MUTED, marker_chars=6):
    """幂等的小字限定语：已存在同前缀的文本框就更新文字，否则新建（不会重复插入）。"""
    marker = value[:marker_chars]
    for sh in slide.shapes:
        for sub in iter_shapes(sh):
            if getattr(sub, "has_text_frame", False) and sub.text.startswith(marker):
                if sub.text != value:
                    set_text(sub.text_frame, value)
                return sub
    return text(slide, value, x, y, w, h, size, color)


def set_notes(slide, value):
    slide.notes_slide.notes_text_frame.text = value


# ----------------------------------------------------------------- 正文替换表
# 顺序敏感：长句 / 具体句替换在前，宽泛的数值替换在后（否则短规则会先把长句截断）。
REPLACEMENTS = [
    # --- 验收项条数：与 python tools/selfcheck.py 的实际 @check 数（34）一致（t18/V2）---
    # 只替换「37 条验收项」这一整串，避免误伤页面上的 Pet-37 / 37 类 / 37 维 / 37.5%。
    ("37 条验收项", "34 条验收项"),
    # --- P04 来源补全：MACs 列混用两个来源，页脚必须把 family_summary.csv 也列上（t18/V5）---
    ("数据来源：outputs/pretrained_eval/<model>/metrics.json · summary.csv",
     "数据来源：outputs/pretrained_eval/<model>/metrics.json · outputs/benchmarks/family_summary.csv（M1.1+ 的 MACs）· summary.csv"),
    # --- 归一化：把早期中间态的写法收敛到当前文案（替换是单向的，这一步保证重复运行仍正确）---
    ("6 个 ONNX · ORT CPU（一致性只测了 3 个型号）", "6 个 ONNX · ORT CPU"),
    ("P50 7.1 ~ 32.7 ms（延迟，非误差）", "P50 7.1 ~ 32.7 ms"),
    ("max|Δ| 7.093e-06（32 个随机输入）· 32/32 一致", "max|Δ| 7.093e-06 · 32/32 一致"),
    ("ONNX Top-1 一致（3 型号各 12 张）率只测了", "PyTorch 与 ONNX 的 Top-1 一致率只测了"),
    # --- D4 外推：单型号一致率不能说成「六个模型一致」，九类问题不能说成已排除 ---
    ("6 个 ONNX 模型，PyTorch↔ONNX Top-1 一致率 100%",
     "6 个 ONNX 模型；一致性只测了 3 个型号各 12 张"),
    ("重参数化数值等价、BN 归零；ONNX 一致率 100%",
     "重参数化 32 个随机输入通过（max 7.093e-06）；ONNX 一致性测了 3 个型号各 12 张"),
    ("重参数化数值等价、BN 归零；ONNX 一致率 100", "重参数化 32 个随机输入通过；ONNX 固定 n=12 样本一致"),
    ("没有出现题目列出的九类典型问题",
     "这 12 张图上未观察到九类典型问题的表现（不等于已排除九类问题）"),
    # --- D3 封面 / 状态页：把「重参数化误差」和「ONNX 一致率」拆开口径并加限定语 ---
    # 注意：状态页卡片文本框只有 ~2.5in 宽，卡片内文案必须短，长限定语放在页脚那一行。
    ("重参数化 max|Δlogits|", "结构重参数化（32 随机输入）max|Δ|"),
    ("max|Δ| 7.1e-06 · Top-1 32/32", "max|Δ| 7.093e-06 · 32/32 一致"),
    ("PyTorch↔ONNX Top-1 一致", "ONNX Top-1 一致（3 型号各 12 张）"),
    # --- 展示值统一 4 位有效数字（D3/D7）---
    ("7.1e-06", "7.093e-06"),
    # --- D5 超证据断言：不再说「完全等价」---
    ("重参数化在数值上完全等价", "重参数化误差小于设定阈值"),
    ("结构重参数化在数值上完全等价", "结构重参数化误差小于设定阈值"),
    ("代数上等价；FP32 运算顺序改变会产生舍入误差。这里通过的是已测输入和设定阈值。",
     "做法：把 BN 的均值方差折进卷积核，再把 3×3、1×1 与 identity 三条分支的核相加，必须在 eval() 之后做。\n"
     "看到：32 个固定随机输入 max|Δlogits| = 7.093e-06（< 1e-4 阈值），Top-1 32/32 一致，BN 模块 107 → 0。\n"
     "还不能说明：这是代数等价的替换 + 已测输入 + 设定阈值，不是位级完全相等，也不代表所有输入都如此。"),
    # --- 训练/优化页的学术化措辞 ---
    ("八个实验都落在 0.6 个点的窄带内 —— 这是噪声，不是提升", "8 组结果差距很小，尚不能确认稳定提升"),
    ("8 组实验落在 0.6 个点的窄带内 —— 这是噪声，不是提升", "8 组结果差距很小，尚不能确认稳定提升"),
    ("四项优化均未超出噪声：这本身是有价值的负面结论", "四项优化未显示稳定提升；需要多种子实验进一步判断"),
    # --- 学术化措辞改直白（怎么做 / 看到了什么 / 还不能说明什么）---
    ("合规性自证", "训练检查"),
    ("软标签平滑决策面，等价于训练集扩容", "混合两张图片与标签，尝试减少对训练样本的记忆"),
    ("压缩纹理记忆", "减少对局部纹理的依赖"),
    ("主干锚定在预训练解附近", "让主干更新幅度更小"),
    ("实测证明两条路线权重逐位等价（max|Δlogits| = 0），省去手写键转换器",
     "实测两条路线权重逐位相同（同进程同权重对比，max|Δlogits| = 0），省去手写键转换器"),
    ("BN 108 → 0", "BN 107 → 0"),
    ("五曲线 · 混淆矩阵 · Grad-CAM", "五条曲线 · 混淆矩阵 · Grad-CAM"),
    ("官方权重实测复现到 0.5 个点以内", "官方权重在自建子集上的实际表现"),
    ("每个数字都能回到 outputs/ 里的文件", "关键数字与实验口径见 outputs/ 和审计说明"),
    ("重参数化数值等价", "重参数化 32 个随机输入通过"),
]


def replace_in_shape(shape, pairs):
    if getattr(shape, "has_text_frame", False):
        for p in shape.text_frame.paragraphs:
            for old, new in pairs:
                if any(old in r.text for r in p.runs):
                    for r in p.runs:
                        r.text = r.text.replace(old, new)
                elif old in p.text and p.runs:      # 整句被拆进多个 run：保留首个 run 的字体
                    p.runs[0].text = p.text.replace(old, new)
                    for r in p.runs[1:]:
                        r.text = ""
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            for cell in row.cells:
                for p in cell.text_frame.paragraphs:
                    for old, new in pairs:
                        if old in p.text and p.runs:
                            p.runs[0].text = p.text.replace(old, new)
                            for r in p.runs[1:]:
                                r.text = ""
    if getattr(shape, "shapes", None) is not None:
        for child in shape.shapes:
            replace_in_shape(child, pairs)


def apply_replacements(prs):
    for s in prs.slides:
        for sh in s.shapes:
            replace_in_shape(sh, REPLACEMENTS)
        if s.has_notes_slide:
            for old, new in REPLACEMENTS:
                tf = s.notes_slide.notes_text_frame
                if old in tf.text:
                    tf.text = tf.text.replace(old, new)


def renumber(prs):
    """页脚两位页码按最终顺序重排；只认「贴底 + 很扁」的页脚组，避免误改卡片编号。"""
    for i, s in enumerate(prs.slides, 1):
        want = f"{i:02d}"
        for sh in s.shapes:
            if sh.shape_type == MSO_SHAPE_TYPE.TEXT_BOX and re.fullmatch(r"\d{2}", sh.text.strip() or ""):
                set_text(sh.text_frame, want)
            elif sh.shape_type == MSO_SHAPE_TYPE.GROUP and sh.top is not None:
                if sh.top / EMU > 6.4 and sh.height / EMU < 0.6:
                    for sub in iter_shapes(sh):
                        if getattr(sub, "has_text_frame", False) and re.fullmatch(r"\d{2}", sub.text.strip() or ""):
                            set_text(sub.text_frame, want)


# --------------------------------------------------------------------- 页面构建
def merge_and_place_slides(prs):
    """P03（结构）+ P04（为什么不是 ViT）合并为一页，P04 腾出的位置给「预测结果」页。

    做法上不新增也不删除幻灯片部件，只搬动一张已有幻灯片并在原位重建：
    首次运行把原 P04 挪到第 11 页再重建为「预测结果」页；重复运行时两页都按标题找到。
    这样总页数始终 ≤15，也不会产生重复的 ``ppt/slides/slideN.xml`` 部件名。
    """
    merged = find_slide(prs, STRUCT_TITLE, required=False)
    pred = find_slide(prs, PRED_MARK, required=False)
    if merged is None and pred is None:                 # 首次运行：合并 + 搬页
        _, merged = find_slide_pos(prs, STRUCT_MARK)
        idx, pred = find_slide_pos(prs, NOT_VIT_MARK)
        move_slide(prs, idx, 10)
    if merged is None:
        merged = find_slide(prs, STRUCT_MARK)
    if pred is None:
        raise RuntimeError("找不到可复用为「预测结果」页的幻灯片")
    return merged, pred


MERGE_NOTES = (
    "【本页是合并页】试题第 16–17 页建议结构的「环节 2 RepViT 核心结构」与「环节 3 为什么 RepViT 不是标准 ViT」"
    "合并到同一页：左栏 = 环节 2，右栏 = 环节 3。合并原因：PPT 需控制在 15 页内，"
    "同时新增第 11 页「预测结果」承载试题第 10 页的三条可视化硬要求（≥8 张测试集预测 / ≥4 张同图对比 / ≥5 张训练集外实际图片）。\n"
    "题干 15 个建议环节的落点：1 题目与任务完成情况→P01–P02；2 RepViT 核心结构→P03 左栏；3 为什么不是标准 ViT→P03 右栏；"
    "4 官方多型号评价→P04；5 数据集和训练流程→P05；6 Baseline 结果→P06；7 优化假设与控制变量→P07；"
    "8 曲线和定量结果→P08；9 混淆矩阵与失败案例→P09；10 Grad-CAM 结果→P10；11 结构重参数化→P12；"
    "12 ONNX 多模型部署→P13；13 性能比较→P14；14 遇到的问题→P15；15 总结→P15。"
    "（P11 为新增页，对应试题第 10 页「必须提供的结果」里的预测可视化三条。）\n"
    "数据来源：python tools/draw_arch.py → outputs/architecture/repvit_m0_9_arch.png / .md；"
    "ONNX 算子直方图见 outputs/reparam/repvit_m0_9_onnx_nodes.json。形状由 forward hook 现测，非论文插图。"
)


def build_structure_slide(s):
    clear_slide(
        s, STRUCT_TITLE,
        "环节 2 + 环节 3 合并页 · 左：M0.9 结构（形状由 forward hook 现测）；右：标准 ViT 与 RepViT 逐项对应",
        3, "数据来源：outputs/architecture/repvit_m0_9_arch.md/.png（python tools/draw_arch.py）· outputs/reparam/repvit_m0_9_onnx_nodes.json",
        MERGE_NOTES, title_size=25)

    text(s, "左栏 · 环节 2：M0.9 是一条纯卷积流水线", .6, 1.62, 5.85, .3, 16, CYAN, True)
    text(s, "输入　224×224×3 RGB\n"
            "Stem　conv1 3×3 s2 → 24ch 112×112；conv2 3×3 s2 → 48ch 56×56\n"
            "Stage 0　2 × RepViT Block　48ch 56×56\n"
            "Stage 1　2 × RepViT Block　96ch 28×28\n"
            "Stage 2　14 × RepViT Block　192ch 14×14（Block 最多）\n"
            "Stage 3　2 × RepViT Block　384ch 7×7\n"
            "GAP → 384 维　Head RepVitClassifier（NormLinear）→ 37 维",
         .6, 1.98, 5.85, 2.3, 13)
    text(s, "单个 RepViT Block 内部", .6, 4.34, 5.85, .3, 15, CYAN, True)
    text(s, "空间混合（Token Mixer）：RepVGGDW = 3×3 dw + 1×1 dw\n"
            "通道混合（Channel Mixer）：1×1 升维 → GELU → 1×1 降维\n"
            "残差连接；SE 只放在部分 Block（全局池化有真实延迟）\n"
            "训练态：3×3 dw 分支 + 1×1 dw 分支，各自带 BatchNorm\n"
            "推理态：BN 折进卷积核后相加 → 单个 3×3 depthwise 卷积",
         .6, 4.68, 5.85, 1.35, 13)
    text(s, "一句话：整条链路只有卷积 / GELU / 池化，没有注意力；名字里的 ViT 说的是宏观架构选择。",
         .6, 6.2, 5.85, .6, 13, GREEN)

    text(s, "右栏 · 环节 3：标准 ViT 与 RepViT 的对应关系", 6.65, 1.62, 6.15, .3, 16, CYAN, True)
    table(s, [["环节", "标准 ViT", "RepViT"],
              ["空间混合", "Multi-Head Self-Attention", "RepVGGDW（3×3 dw + 1×1 dw）"],
              ["通道混合", "MLP / FFN", "Channel Mixer（1×1→GELU）"],
              ["下采样", "patchify（stride=16 大核）", "Early Conv Stem（两组 stride=2）"],
              ["归纳偏置", "弱，依赖大数据", "强（卷积自带局部性 / 平移等变）"],
              ["ONNX 图内", "MatMul + Softmax", "无注意力算子"]],
          6.65, 1.96, 6.15, 2.2, sizes=[1.0, 2.25, 2.9], font=10)
    text(s, "实证：onnx/*.onnx 的算子直方图里只有 Conv / Gemm / Add / Mul / Div / Clip / Relu，"
            "没有 MatMul + Softmax 这一对注意力算子组合（outputs/reparam/repvit_m0_9_onnx_nodes.json）。",
         6.65, 4.5, 6.15, .62, 11)
    text(s, "四个关键设计", 6.65, 5.28, 6.15, .3, 15, CYAN, True)
    text(s, "① 分离 Token / Channel Mixer —— 空间与通道容量可独立调节\n"
            "② 扩张比降到约 2 并同时加宽 —— 省下的 MACs 换成通道数\n"
            "③ SE 不做每块都放 —— 全局池化的延迟不可忽略\n"
            "④ 简单分类头 —— 单个 BN + Linear，省真实延迟\n"
            "M0.9 实测：20 个 Block，按 Stage 分布 [2, 2, 14, 2]",
         6.65, 5.62, 6.15, 1.3, 12)


def build_pred_slide(s, note_cmd):
    b8 = rows("outputs/predictions/baseline_test_preds.csv")
    bo = rows("outputs/predictions/opt_combo_test_preds.csv")
    grid8 = b8[:8]
    n_ok8 = sum(int(r["correct"]) for r in grid8)
    p8 = [float(r["prob"]) for r in grid8]
    cmpj = data("outputs/predictions/baseline_vs_opt_combo_predict_compare.json")
    csum = cmpj["summary"]
    opt_by_id = {r["image_id"]: r for r in bo}
    n_diff = sum(1 for r in b8 if r["image_id"] in opt_by_id
                 and r["pred_idx"] != opt_by_id[r["image_id"]]["pred_idx"])
    ext = rows("outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv")
    # 出图用的 5 张 = `--images external/*.JPEG --num 5`，即按文件名排序的前 5 张（不是 CSV 前 5 行）
    ext_by_name = {Path(r["path"]).name: r for r in ext}
    ext5 = [ext_by_name[p.name] for p in sorted((ROOT / "external").glob("*.JPEG"))[:5]]
    ext_p = "、".join(f"{r['pred_name']} {float(r['prob']):.3f}" for r in ext5)
    ext_lo = min(ext5, key=lambda r: float(r["prob"]))
    dist = distribution_stats()
    ds, ex = dist["dataset"], dist["external"]

    clear_slide(
        s, PRED_TITLE,
        "Baseline = RepViT-M0.9 Pet-37（test Top-1 92.34%）· 逐张 Top-5 类别与置信度 · 每个面板下方标注该面板自己的推理口径",
        11, "来源：outputs/predictions/{baseline_test_preds,cases_test_baseline}.csv · baseline_vs_opt_combo_predict_compare.json · "
            "outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv · distribution_compare_summary.csv",
        note_cmd)

    # 左栏：Baseline vs 组合方案同一批图片
    text(s, "① 同图对比 4 组\nBaseline vs 组合方案", .55, 1.6, 3.3, .5, 13, CYAN, True)
    picture(s, "outputs/predictions/compare_baseline_vs_opt_combo_grid4.png", .55, 2.14, 3.3, 3.1)
    text(s, f"按 image_id 配对（不按行号）；全测试集里两模型预测不同的有 {n_diff} 张。\n"
            f"落盘 JSON 另抽的 4 张两组都判对（both_correct={csum['both_correct']}、"
            f"fixed_by_opt={csum['fixed_by_opt']}、broken_by_opt={csum['broken_by_opt']}）——"
            f"同图对比要看有差异的样本。\n"
            f"口径：PyTorch / CUDA / FP32 / 224×224 / eval batch 128 / 每张只推理一次。",
         .55, 5.34, 3.3, 1.6, 10.5, MUTED)

    # 右上：测试集预测 8 张
    text(s, "② 测试集预测 8 张（Top-5 类别 + 置信度）", 3.95, 1.6, 8.85, .3, 15, CYAN, True)
    picture(s, "outputs/predictions/test_top5_baseline_grid8.png", 3.95, 1.9, 8.85, 1.7)
    text(s, f"取测试列表顺序前 8 张（{grid8[0]['image_id']} … {grid8[-1]['image_id']}）："
            f"{n_ok8}/8 判对，Top-1 置信度 {min(p8):.3f} ~ {max(p8):.3f}；顺序取样，不是挑出来的好例子。\n"
            f"口径：PyTorch / CUDA / FP32 / 224×224 / eval batch 128 / 每张只推理一次。",
         3.95, 3.64, 8.85, .5, 10.5, MUTED)

    # 右下：训练集以外的实拍图片 5 张
    text(s, "③ 训练集以外的实拍图片 5 张（Top-5 类别 + 置信度）", 3.95, 4.2, 8.85, .3, 15, CYAN, True)
    picture(s, "outputs/predictions/external_top5_pet37_grid5.png", 3.95, 4.5, 8.85, .75)
    text(s, f"Top-1 置信度（图从左到右）：{ext_p} —— 5 张都判到对应品种。\n"
            f"最低的 {ext_lo['pred_name']} 只有 {float(ext_lo['prob']):.3f}：置信度低不等于判错，反过来高置信度也不保证一定可靠。\n"
            f"外部实拍共 {len(ext)} 张，本页展示按文件名排序的前 5 张（试题要求 ≥5 张）；没有 GT（真值），只看 Top-5 与置信度，不报准确率。\n"
            f"数据集 vs 外部图片的分布差异：短边均值 {ds['short_side']:.0f} → {ex['short_side']:.0f} px，"
            f"亮度均值 {ds['brightness_mean']:.3f} → {ex['brightness_mean']:.3f}，"
            f"背景复杂度 {ds['bg_complexity_ratio']:.3f} → {ex['bg_complexity_ratio']:.3f}；分布不同，不能直接比准确率。\n"
            f"口径：ONNX Runtime 1.30.0 CPUExecutionProvider / FP32 / batch 1 / 224×224 / i7-13650HX / 每张只推理一次。",
         3.95, 5.3, 8.85, 1.65, 10.5, MUTED)


def build_curves_slide(s, n_eval_base, n_eval_opt):
    baseline = rows("outputs/logs/baseline_metrics.csv")
    opt = rows("outputs/logs/opt_combo_metrics.csv")
    clear_slide(
        s, "训练曲线：两组都收敛，组合方案未提高测试准确率",
        f"RepViT-M0.9 · Pet-37 · 输入 224×224 · batch 64 · NVIDIA GeForce RTX 4060 Laptop GPU · "
        f"PyTorch 2.14.0+cu126（bf16 autocast）· 曲线共用坐标范围 · test 评价次数 Baseline {n_eval_base} / 组合 {n_eval_opt}",
        8, "来源：outputs/logs/{baseline,opt_combo}_metrics.csv；outputs/metrics/{baseline,opt_combo}_test.json；"
           "outputs/logs/*_test_eval_count.json",
        "python tools/plot_curves.py --runs baseline=outputs/logs/baseline_metrics.csv opt_combo=outputs/logs/opt_combo_metrics.csv --out outputs/curves/opt_compare.png\n"
        "元信息：模型 RepViT-M0.9（timm 实现，num_classes=37）；输入 224×224；训练 batch 64 / 评价 batch 128；"
        "硬件 RTX 4060 Laptop GPU（CUDA 12.6）；推理后端 PyTorch 2.14.0+cu126；精度 bf16 autocast（评价 FP32）；"
        "测试集 3669 张，评价次数 Baseline=1、组合=2（落盘 eval_count）。\n"
        "曲线直接来自 CSV；epoch 从 0 开始。Mixup/CutMix、Label Smoothing 影响 train loss，不直接把训练损失高低当作泛化优劣。\n"
        "单种子、同一测试集上的小差异不能证明统计显著或稳定提升。")
    keys = [("train_loss", "训练损失", 1, 4, "0.0"), ("val_loss", "验证损失", 1, 4, "0.0"),
            ("val_top1", "验证 Top-1 (%)", 100, 100, "0"), ("val_macro_f1", "验证 Macro-F1 (%)", 100, 100, "0"),
            ("lr", "学习率", 1, .0011, "0.0000")]
    for i, (key, title, mul, hi, fmt) in enumerate(keys):
        x, y = .6 + (i % 3) * 4.16, 1.72 + (i // 3) * 2.43
        chart(s, title, [str(r["epoch"]) for r in baseline],
              [("Baseline", [float(r[key]) * mul for r in baseline]),
               ("opt_combo", [float(r[key]) * mul for r in opt])],
              x, y, 3.98, 2.2, lo=0, hi=hi, fmt=fmt)
    bt, ot = data("outputs/metrics/baseline_test.json"), data("outputs/metrics/opt_combo_test.json")
    text(s, "测试集结果（3669 张）", 8.93, 4.15, 3.8, .35, 16, CYAN, True)
    table(s, [["指标", "Baseline", "组合"], ["Top-1", f"{bt['top1']*100:.2f}%", f"{ot['top1']*100:.2f}%"],
              ["Top-5", f"{bt['top5']*100:.2f}%", f"{ot['top5']*100:.2f}%"],
              ["Macro-F1", f"{bt['macro_f1']*100:.2f}%", f"{ot['macro_f1']*100:.2f}%"]],
          8.93, 4.59, 3.8, 1.35, font=11)
    text(s, "前几轮提升最快，后段逐渐稳定。\n软标签改变 train loss，不能直接横比。\n单种子结果不足以判断稳定提升。",
         8.93, 6.1, 3.8, .78, 12, MUTED)


def build_cm_slide(s):
    cb = data("outputs/confusion_matrix/baseline_cat_dog_block.json")
    bt = data("outputs/metrics/baseline_test.json")
    pc = rows("outputs/confusion_matrix/baseline_per_class.csv")
    worst = min(pc, key=lambda r: float(r["f1"]))
    best = max(pc, key=lambda r: float(r["f1"]))
    clear_slide(
        s, "错在哪里：多数是同物种内的品种混淆",
        "Baseline · Pet-37 test · 3669 张 · PyTorch 2.14.0+cu126 / CUDA / FP32 / 224×224 / 每张只推理一次 · "
        "左：行归一化混淆矩阵；中：每类 F1 柱状图；右：失败案例",
        9, "来源：outputs/confusion_matrix/{baseline_cm.png,baseline_per_class.csv,baseline_per_class_f1.png,baseline_cat_dog_block.json}；"
           "outputs/predictions/case_wrong_baseline_case*.png",
        "python tools/check_cm.py --pred-csv outputs/predictions/baseline_test_preds.csv --classes labels/pet_classes.txt --out-dir outputs/confusion_matrix\n"
        "左图按真实类别逐行归一化；中间柱状图是 37 类的每类 F1（数据源 outputs/confusion_matrix/baseline_per_class.csv）。\n"
        "正确的案例见第 11 页（8 张测试集预测）与第 10 页 Grad-CAM；本页只保留 2 个典型失败案例，不回避失败。")
    picture(s, "outputs/confusion_matrix/baseline_cm.png", .6, 1.72, 4.35, 4.43)
    picture(s, "outputs/confusion_matrix/baseline_per_class_f1.png", 5.1, 1.72, 3.0, 4.43)
    text(s, f"{cb['n_error']} 个错误中，{cb['n_cross_species_error']} 个跨猫狗物种（{cb['cross_species_error_ratio']*100:.2f}%），"
            f"其余 {cb['n_within_species_error']} 个是同物种品种判断错误。",
         8.4, 1.75, 4.4, .8, 16, CYAN, True)
    picture(s, "outputs/predictions/case_wrong_baseline_case01.png", 8.4, 2.72, 2.1, 1.85)
    picture(s, "outputs/predictions/case_wrong_baseline_case02.png", 10.65, 2.72, 2.1, 1.85)
    text(s, "上面两张是失败案例：相似外形、遮挡和背景都会影响判断；热力图只能用来检查线索，不能仅凭它确定因果。",
         8.4, 4.72, 4.4, .8, 12.5)
    text(s, f"每类 F1：最低 {float(worst['f1']):.3f}（{worst['class_name']}），最高 {float(best['f1']):.3f}（{best['class_name']}）；"
            f"37 类 Macro-F1 = {bt['macro_f1']*100:.2f}%（数据源 baseline_per_class.csv）。",
         5.1, 6.2, 3.2, .8, 10, GREEN)
    text(s, "下一步：补充容易混淆的品种（上面这类）和不同拍摄角度。", 8.4, 6.5, 4.4, .4, 12, GREEN)


def build_gradcam_slide(s):
    clear_slide(
        s, "Grad-CAM：同时看判对和判错的图片",
        "Baseline · 挂载 stages[-1].blocks[-1] · 原图 / 热力图 / 叠加图 · PyTorch 2.14.0+cu126 / CUDA / FP32 / 224×224 / 每张只推理一次",
        10, "来源：outputs/gradcam/gradcam_{correct,wrong}_baseline_*.png；实拍图片的完整 Top-5 见第 11 页"
            "（outputs/predictions/external_top5_pet37_grid5.png）",
        "python tools/gradcam.py --help\n挂载点与原始图生成配置见 outputs/gradcam/ 下的 JSON。\n"
        "外部实拍样本来自 ImageNet 跨集合图片，8/8 判对只是小样本观察，不能证明跨域泛化或没有过拟合。")
    cams = [("正确：Bengal", "gradcam_correct_baseline_Bengal_30_correct.png"),
            ("正确：Russian Blue", "gradcam_correct_baseline_Russian_Blue_205_correct.png"),
            ("失败：Boxer", "gradcam_wrong_baseline_boxer_2_wrong.png"),
            ("失败：Egyptian Mau", "gradcam_wrong_baseline_Egyptian_Mau_204_wrong.png")]
    for i, (title, fn) in enumerate(cams):
        x, y = .65 + (i % 2) * 6.35, 1.73 + (i // 2) * 2.1
        text(s, title, x, y, 5.8, .3, 15, GREEN if i < 2 else ORANGE, True)
        picture(s, "outputs/gradcam/" + fn, x, y + .38, 5.95, 1.52)
    text(s, "关注到主体 ≠ 品种一定判断正确。热力图用于检查线索，仍要结合失败案例和完整测试集（3669 张）。",
         .65, 6.28, 12, .55, 18)


def build_reparam_slide(s):
    rep = data(B4_REPORT)
    clear_slide(
        s, "重参数化：合并分支后，误差仍在设定阈值内",
        f"RepViT-M0.9 Pet-37（baseline_best.pt）· 1×3×224×224 · batch 8 · n=32 · i7-13650HX CPU · "
        f"PyTorch 2.14.0+cu126 / FP32 · 1 次融合前后对比 · seed=20240912",
        12, f"来源：{B4_REPORT}；本页是 32 个固定随机输入的重参数化实验，与第 13 页 n=12 真实图片的 ONNX 实验分开",
        "python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37\n"
        f"元信息：模型 RepViT-M0.9（num_classes=37）；输入 1×3×224×224；batch 8；n=32 个 torch.randn 固定输入（seed=20240912）；"
        "硬件 i7-13650HX / Windows 11；后端 PyTorch 2.14.0+cu126（CPU 侧 FP32 对比）；1 次完整对比。\n"
        f"max_abs_err={rep['diff']['max_abs_err']}；mean_abs_err={rep['diff']['mean_abs_err']}。32 个输入是 torch.randn，不是真实测试图片。\n"
        "权重 missing=0 / unexpected=0。BN 模块 107→0；ONNX 图的 BN 节点 24→0 是另一种统计，不可混用。\n"
        "历史官方 C=1000 报告存在权重键不匹配，不纳入结论；见 report/LOGITS_AUDIT.md。")
    chart(s, "PyTorch 模块数量", ["BatchNorm", "Conv2d"],
          [("融合前", [rep["before"]["n_bn"], rep["before"]["n_conv2d"]]),
           ("融合后", [rep["after"]["n_bn"], rep["after"]["n_conv2d"]])],
          .65, 1.8, 6, 3.55, XL_CHART_TYPE.COLUMN_CLUSTERED, lo=0, hi=140, fmt="0")
    table(s, [["数值验证（32 个固定随机输入）", "结果"],
              ["最大 |Δlogits|", f"{rep['diff']['max_abs_err']:.3e}"],
              ["平均 |Δlogits|", f"{rep['diff']['mean_abs_err']:.3e}"],
              ["Top-1 一致", "32 / 32"],
              ["Top-5 最小交集", "5 / 5"],
              ["阈值 / 判定", "max < 1e-4 / PASS"]],
          7, 1.95, 5.6, 3.25, sizes=[3.3, 2.3], font=17)
    text(s, "做法：把 BN 的均值方差折进卷积核，再把 3×3、1×1 与 identity 三条分支的核相加，必须在 eval() 之后做。",
         .65, 5.6, 12, .4, 17)
    text(s, f"看到：32 个固定随机输入 max|Δlogits| = {rep['diff']['max_abs_err']:.3e}（< 1e-4 阈值），"
            f"Top-1 32/32 一致，BN 模块 107 → 0。\n"
            f"还不能说明：这是代数等价的替换 + 已测输入 + 设定阈值，不是位级完全相等，也不代表所有输入都如此。",
         .65, 6.06, 12, .9, 14, MUTED)


def build_onnx_slide(s):
    clear_slide(
        s, "PyTorch ↔ ONNX：固定 n=12，比较同一输入张量",
        "原模型 vs 融合后 ONNX · ORT 1.30.0 CPUExecutionProvider · FP32 · batch 1 · 1×3×224×224 · "
        "i7-13650HX / Windows 11 · 每个型号各 12 张真实图片 · 每张只跑一次",
        13, "来源：outputs/metrics/consistency_{repvit_m0_9_pet37,repvit_m0_9_in1k,repvit_m1_0_in1k}.json",
        "python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json\n"
        "元信息：输入 1×3×224×224；batch 1；硬件 i7-13650HX / Windows 11；后端 ONNX Runtime 1.30.0 CPUExecutionProvider；"
        "精度 FP32；每个型号各 12 张真实图片、每张只跑一次。\n"
        "Pet-37 顺序取固定列表前 12 张，每张预处理一次，同一 numpy 张量送入两端。\n"
        "max=6.198883056640625e-06；mean=1.5006899711048998e-06；Top-1/Top-5 集合均 1.0。\n"
        "5.25e-06 是未找到对应完整产物的旧引用；旧命令写 limit=8 不能证明该值来自 n=8。当前权重跑 n=8 也不能复现旧值。\n"
        "不是全测试集一致率，不验证两套预处理独立实现等价，也不能排除所有部署错误；只有 3 个型号各 12 张，不代表全部六个 ONNX 型号。")
    cells = [["模型 / 输入列表", "n", "最大 |Δlogits|", "平均 |Δlogits|", "Top-1 / Top-5"]]
    for key, label in [("repvit_m0_9_pet37", "M0.9 Pet-37 / pet_test"),
                       ("repvit_m0_9_in1k", "M0.9 / ImageNet 子集"), ("repvit_m1_0_in1k", "M1.0 / ImageNet 子集")]:
        d = data("outputs/metrics/consistency_" + key + ".json")
        # 展示口径：4 位有效数字（:.3e），与 report/LOGITS_AUDIT_FINDINGS.md 第 1 节一致
        cells.append([label, d["n"], f"{d['max_abs_logits']:.3e}", f"{d['mean_abs_logits']:.3e}", "100% / 100%"])
    table(s, cells, .65, 1.9, 12, 2.15, sizes=[3.65, .6, 2.4, 2.4, 2.95], font=16)
    text(s, "1  重参数化 7.093e-06（32 个随机输入）是第 12 页的另一批实验，不与本页并成一句结论。\n"
            "2  部署预处理由 PIL 独立实现；本次对比把同一张量送入两个推理后端，因此这里不比预处理实现。\n"
            "3  结论仅覆盖这 12 张图片：不能说成完整测试集，也不能说成全部六个模型一致。\n"
            "4  旧的 5.25e-06 缺少对应完整记录，当前统一引用上表落盘值（4 位有效数字）。",
         .7, 4.35, 12, 1.6, 18)
    text(s, "复跑：python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 12\n"
            "完整命令、固定列表与输出路径已写入本页备注和 report/LOGITS_AUDIT.md。",
         .7, 6.12, 12, .73, 15, GREEN)


def model_label(key):
    if key.endswith("_pet37"):
        return key.replace("repvit_", "").replace("_pet37", "").replace("m", "M").replace("_", ".") + "-Pet37"
    return key.replace("repvit_", "").replace("_in1k", "").replace("m", "M").replace("_", ".")


def build_bench_slide(s):
    br = rows("outputs/benchmarks/summary.csv")
    b0 = br[0]
    labels = [model_label(r["model"]) for r in br]
    imnet = [r for r in br if r["model"].endswith("_in1k")]
    acc = {}
    for r in imnet:
        arch = r["model"].replace("_in1k", "")
        d = data("outputs/pretrained_eval/" + arch + "/metrics.json")
        a = d.get("top1", d.get("accuracy_top1"))
        if a is None:
            raise KeyError(f"missing top1: {arch}")
        acc[r["model"]] = (float(a), float(d["batch_size"]), d["device"], d["precision"])
    cpu = b0["cpu_model"].split(")")[-1].strip()
    provider = re.sub(r"[\[\]']", "", b0["provider_actual"])
    clear_slide(
        s, "部署怎么选：先看同一机器上的延迟和准确率",
        f"型号：{ ' / '.join(labels) } · 输入 1×3×224×224 · batch {b0['batch_size']} · {cpu} · "
        f"ORT {b0['ort_version']} {provider} · {b0['precision']} · "
        f"预热 {b0['warmup']} + 正式 {b0['runs']} 次 · threads {b0['threads_intra']}",
        14, "来源：outputs/benchmarks/summary.csv（含 cpu_model / os / ort_version / warmup / runs / threads 全字段）；"
            "outputs/pretrained_eval/<model>/metrics.json",
        "python deploy/benchmark.py --model repvit_m0_9_in1k --model repvit_m1_0_in1k --model repvit_m0_9_pet37 --warmup 10 --runs 50 --threads 4 --out-dir outputs/verification/benchmarks\n"
        f"元信息：6 个 ONNX 模型（{ ' / '.join(labels) }）；输入 1×3×224×224；batch 1；硬件 {b0['cpu_model']} / {b0['os']}；"
        f"后端 ONNX Runtime {b0['ort_version']} {provider}；精度 {b0['precision']}；"
        f"每个模型预热 {b0['warmup']} 次 + 正式 {b0['runs']} 次（threads={b0['threads_intra']}）。\n"
        "左图含六个部署模型，只比速度；右图只放同一 ImageNet 子集的五个型号（准确率来自 PyTorch，延迟来自 ONNX Runtime CPU）。\n"
        "官方 iPhone 延迟与本机 CPU 延迟不可直接比较。性能数据是历史落盘值，现场新测会有波动。")
    chart(s, "六个 ONNX 模型的推理耗时 (ms)", labels,
          [("P50", [float(r["p50_ms"]) for r in br]), ("P95", [float(r["p95_ms"]) for r in br])],
          .65, 1.85, 6.05, 3.95, XL_CHART_TYPE.COLUMN_CLUSTERED, lo=0, hi=36, fmt="0")
    text(s, "同一 ImageNet 子集（1000 张）：速度与准确率", 7.05, 1.85, 5.6, .3, 16, CYAN, True)
    xy = XyChartData()
    for r in imnet:
        se = xy.add_series(model_label(r["model"]))
        se.add_data_point(float(r["p50_ms"]), acc[r["model"]][0])
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.05), Inches(2.2), Inches(5.6), Inches(3.55))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor.from_string("FFFFFF"); bg.line.fill.background()
    ch = s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(7.05), Inches(2.2), Inches(5.6), Inches(3.55), xy).chart
    ch.has_legend = True
    ch.legend.position = XL_LEGEND_POSITION.BOTTOM
    ch.legend.font.size = Pt(11)
    ch.font.size = Pt(11)
    ch.category_axis.has_title = True
    ch.category_axis.axis_title.text_frame.text = "P50 (ms)"
    ch.value_axis.has_title = True
    ch.value_axis.axis_title.text_frame.text = "Top-1 (%)"
    ch.value_axis.minimum_scale = 77
    ch.value_axis.maximum_scale = 84
    for se in ch.series:
        se.marker.style = XL_MARKER_STYLE.CIRCLE
        se.marker.size = 10
    # 「Top-1 相同但更快」的一对从落盘值里现算，不写死型号
    pair = None
    for i in range(len(imnet)):
        for j in range(i + 1, len(imnet)):
            if acc[imnet[i]["model"]][0] == acc[imnet[j]["model"]][0]:
                f, sl = (imnet[i], imnet[j]) if float(imnet[i]["p50_ms"]) < float(imnet[j]["p50_ms"]) else (imnet[j], imnet[i])
                pair = (f, sl)
    lead = (f"本子集上 {model_label(pair[0]['model'])} 与 {model_label(pair[1]['model'])} 的 Top-1 相同"
            f"（{acc[pair[0]['model']][0]:.1f}%），{model_label(pair[0]['model'])} 更快"
            f"（P50 {float(pair[0]['p50_ms']):.1f} ms vs {float(pair[1]['p50_ms']):.1f} ms）。"
            if pair else "右图只放同一子集的五个型号，看的是「多花多少毫秒换多少点准确率」。")
    a0 = acc[imnet[0]["model"]]
    text(s, f"{lead}更大模型的额外耗时，需要结合使用场景判断。\n"
            f"右图元信息：准确率来自 PyTorch 2.14.0+cu126（{a0[1]:.0f} batch / {a0[2]} / {a0[3]}）在同一 1000 张子集上的评价；"
            f"延迟来自本机 ONNX Runtime CPU（batch 1）。两者口径不同，只做参考。",
         .65, 5.85, 12, 1.05, 13)


def add_cover_and_status_limits(prs):
    """D3/D4：封面与状态页的口径限定语（幂等）。"""
    cover = prs.slides[0]
    ensure_footnote(
        cover,
        "口径分开：结构重参数化 = 32 个固定随机输入（max|Δ| 7.093e-06，Top-1 32/32）；"
        "ONNX 一致性 = Pet-37 / M0.9 / M1.0 三个型号各 12 张真实图片（第 13 页，3 份 n=12 落盘），"
        "只覆盖这 12 张，不代表全部六个 ONNX 型号。旧的 5.25e-06 未找到配套产物。",
        .98, 6.92, 11.35, .5, 11, MUTED, marker_chars=6)
    set_notes(cover, "封面数字的口径：78.20% = 自建 1000 张 ImageNet 子集（PyTorch / CUDA / batch 64）；"
                     "92.34% = Pet-37 test 3669 张（只评价 1 次）；7.093e-06 = 结构重参数化 B4（32 个固定随机输入，"
                     "outputs/reparam/repvit_m0_9_pet37_reparam_report.json）；100% = PyTorch↔ONNX B1（Pet-37 前 12 张真实图片）。"
                     "重参数化误差与 ONNX 一致率是两批实验，不并成一句结论。\n"
                     "逐条依据与桶分级见 report/LOGITS_AUDIT_FINDINGS.md 第 1 节。")

    status = prs.slides[1]
    ensure_footnote(
        status,
        "口径与边界：PyTorch 与 ONNX 的 Top-1 一致率只测了 Pet-37 / ImageNet-M0.9 / ImageNet-M1.0 三个型号各 12 张（均为 100%）；"
        "其余型号只有导出检查与性能结果，不做同口径一致性声明。重参数化 7.093e-06 来自 32 个固定随机输入，与上面这批 ONNX 实验不是同一批。",
        .74, 7.05, 11.85, .35, 10.5, MUTED, marker_chars=6)
    set_notes(status, "六个环节卡片对应试题基础任务 1–6 的交付物；卡片里每行末尾是落盘路径。\n"
                      "口径边界（试题第 16–17 页要求区分：论文 / 官方仓库 / 官方权重实跑 / 自训 Baseline / 优化模型 / PyTorch / ONNX）："
                      "卡片 01 是官方权重实跑，02/03/05 是自训结果，05 是 PyTorch 侧重参数化，06 是 ONNX 部署与性能。\n"
                      "ONNX 一致性只有 3 个型号各 n=12 的落盘 JSON，其余型号不做一致性声明（不含 5.25e-06 旧引用，该值无配套产物）。")


# ------------------------------------------------------------------ 内容同步
def write_content(prs):
    out = ["# 答辩 PPT 内容与证据", "",
           "15 页（P03 已把试题建议环节 2「RepViT 核心结构」与环节 3「为什么不是标准 ViT」合并为一页，腾出 P11「预测结果」页）；",
           "图表由 tools/update_defense.py 从落盘 CSV/JSON 填充；PDF 由 tools/export_defense_pdf.py 用 PowerPoint COM 导出。", ""]
    for i, s in enumerate(prs.slides, 1):
        out += [f"## P{i:02d}", ""]
        for sh in s.shapes:
            for sub in iter_shapes(sh):
                if getattr(sub, "has_text_frame", False) and sub.text.strip():
                    out.append(sub.text.replace("\v", "\n"))
                if getattr(sub, "has_table", False):
                    for row in sub.table.rows:
                        out.append(" / ".join(c.text for c in row.cells))
        if s.has_notes_slide:
            out += ["", "备注与验证命令：", s.notes_slide.notes_text_frame.text]
        out += [""]
    (ROOT / "report/PPT_CONTENT.md").write_text("\n\n".join(out), encoding="utf-8")


def eval_count(name):
    d = data(f"outputs/logs/{name}_test_eval_count.json")
    return d["count"]


def main():
    prs = Presentation(PPT)
    trim_tail(prs, 15)

    # 1) P03+P04 合并为一页；原 P04 的位置留给新增的「预测结果」页（不增删页面部件）
    merged, pred = merge_and_place_slides(prs)
    if len(prs.slides) != 15:
        raise RuntimeError(f"页数必须是 15（≤15 硬约束），当前 {len(prs.slides)} 页")

    # 2) 新增页与合并页
    build_structure_slide(merged)
    build_pred_slide(pred, "本页三个面板的出图命令：\n"
                           "python tools/plot_predictions.py --pred-csv outputs/predictions/baseline_test_preds.csv --classes labels/pet_classes.txt --num 8 --cols 4 --out outputs/predictions/test_top5_baseline_grid8.png --tag baseline\n"
                           "python tools/plot_side_by_side.py --pred-csv-a outputs/predictions/baseline_test_preds.csv --pred-csv-b outputs/predictions/opt_combo_test_preds.csv --name-a baseline --name-b opt_combo --num 4 --out outputs/predictions/compare_baseline_vs_opt_combo_grid4.png\n"
                           "python tools/predict_external.py --model repvit_m0_9_pet37 --dir external --num 8  → outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv\n"
                           "python tools/plot_predictions.py --images external/*.JPEG --pred-csv outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv --num 5 --out outputs/predictions/external_top5_pet37_grid5.png --tag pet37\n"
                           "数据源：outputs/predictions/baseline_test_preds.csv（3669 行）、baseline_vs_opt_combo_predict_compare.json、"
                           "pet_compare_ids.json、outputs/benchmarks/external_top5_repvit_m0_9_pet37.csv、distribution_compare_summary.csv。\n"
                           "同图对比按 image_id 做 inner join（不能按行号对齐）；外部图片没有 GT，只看 Top-5 与置信度。")

    # 3) 数据页（按标题定位，避免下标串位）
    build_curves_slide(find_slide(prs, CURVES_MARK), eval_count("baseline"), eval_count("opt_combo"))
    build_cm_slide(find_slide(prs, CM_MARK))
    build_gradcam_slide(find_slide(prs, CAM_MARK))
    build_reparam_slide(find_slide(prs, REPARAM_MARK))
    build_onnx_slide(find_slide(prs, ONNX_MARK))
    build_bench_slide(find_slide(prs, BENCH_MARK))

    # 4) 口径限定语（封面 / 状态页）+ 全页正文与备注的措辞替换
    add_cover_and_status_limits(prs)
    apply_replacements(prs)

    # 5) 页脚页码按最终顺序重排
    renumber(prs)

    prs.save(PPT)
    write_content(prs)
    print(f"Updated {len(prs.slides)} slides (P03+P04 merged, P11 predictions added); "
          f"charts from saved CSV/JSON; notes and PPT_CONTENT synchronized.")


if __name__ == "__main__":
    main()
