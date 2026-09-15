"""Rebuild data-heavy defense slides from saved experiment artifacts (no inference)."""
import csv
import json
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.chart.data import CategoryChartData, XyChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_MARKER_STYLE
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PPT = ROOT / "report/答辩PPT_RepViT.pptx"
BG, WHITE, MUTED, CYAN, GREEN, ORANGE = "101B30", "E6EDF7", "B6C5DC", "55C5EF", "48D9AE", "F2B560"


def data(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def rows(path):
    with (ROOT / path).open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def text(s, value, x, y, w, h, size=17, color=WHITE, bold=False):
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
        p.space_after = Pt(6)
    return sh


def clear_slide(s, title, subtitle, number, source, notes=""):
    for sh in list(s.shapes):
        s.shapes._spTree.remove(sh._element)
    s.background.fill.solid()
    s.background.fill.fore_color.rgb = RGBColor.from_string(BG)
    text(s, title, .6, .35, 12.1, .6, 29, bold=True)
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
    s.shapes.add_picture(str(path), Inches(x + (w-ww)/2), Inches(y+(h-hh)/2),
                         width=Inches(ww), height=Inches(hh))


def chart(s, title, labels, series, x, y, w, h, kind=XL_CHART_TYPE.LINE, lo=None, hi=None, fmt="0.0"):
    text(s, title, x, y, w, .3, 15, CYAN, True)
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y+.34), Inches(w), Inches(h-.34))
    bg.fill.solid(); bg.fill.fore_color.rgb = RGBColor.from_string("FFFFFF")
    bg.line.fill.background()
    cd = CategoryChartData()
    cd.categories = labels
    for name, values in series:
        cd.add_series(name, values)
    ch = s.shapes.add_chart(kind, Inches(x), Inches(y+.34), Inches(w), Inches(h-.34), cd).chart
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
            c = t.cell(i,j)
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


def main():
    prs = Presentation(PPT)
    # Replace the temporary appended curve slide with a chart-rich page 9 in the 15-page narrative.
    while len(prs.slides) > 15:
        node = prs.slides._sldIdLst[-1]
        prs.part.drop_rel(node.rId)
        prs.slides._sldIdLst.remove(node)
    # Preserve existing editable diagrams/layouts; tighten overstatements.
    replacements = {
        "重参数化在数值上完全等价": "重参数化误差小于阈值",
        "结构重参数化在数值上完全等价": "结构重参数化误差小于阈值",
        "8 组实验落在 0.6 个点的窄带内 —— 这是噪声，不是提升": "8 组结果差距很小，尚不能确认稳定提升",
        "四项优化均未超出噪声：这本身是有价值的负面结论": "四项优化未显示稳定提升；需要多种子实验进一步判断",
        "重参数化数值等价、BN 归零；ONNX 一致率 100%": "重参数化 32 个随机输入通过；ONNX 固定 n=12 样本一致",
        "合规性自证": "训练检查",
        "软标签平滑决策面，等价于训练集扩容": "混合两张图片与标签，尝试减少对训练样本的记忆",
        "压缩纹理记忆": "减少对局部纹理的依赖",
        "主干锚定在预训练解附近": "让主干更新幅度更小",
        "BN 108 → 0": "BN 107 → 0",
        "五曲线 · 混淆矩阵 · Grad-CAM": "五条曲线 · 混淆矩阵 · Grad-CAM",
        "官方权重实测复现到 0.5 个点以内": "官方权重在自建子集上的实际表现",
        "每个数字都能回到 outputs/ 里的文件": "关键数字与实验口径见 outputs/ 和审计说明",
    }
    def walk(sh):
        if hasattr(sh, "text_frame"):
            for p in sh.text_frame.paragraphs:
                for r in p.runs:
                    for a,b in replacements.items():
                        r.text = r.text.replace(a,b)
        if hasattr(sh, "shapes"):
            for child in sh.shapes:
                walk(child)
    for s in prs.slides:
        for sh in s.shapes:
            walk(sh)

    baseline, opt = rows("outputs/logs/baseline_metrics.csv"), rows("outputs/logs/opt_combo_metrics.csv")
    s = prs.slides[8]
    clear_slide(s, "训练曲线：两组都收敛，组合方案未提高测试准确率",
                "Pet-37 · Baseline vs opt_combo · 相同划分、seed=42、40 epoch · 每项指标共用坐标范围",
                9, "来源：outputs/logs/{baseline,opt_combo}_metrics.csv；outputs/metrics/{baseline,opt_combo}_test.json",
                "python tools/plot_curves.py --runs baseline=outputs/logs/baseline_metrics.csv opt_combo=outputs/logs/opt_combo_metrics.csv --out outputs/curves/opt_compare.png\n"
                "曲线直接来自 CSV；epoch 从 0 开始。Mixup/CutMix、Label Smoothing 影响 train loss，不直接把训练损失高低当作泛化优劣。\n"
                "学习率来自日志 lr 列。单种子、同一测试集上的小差异不能证明统计显著或稳定提升。")
    keys = [("train_loss","训练损失",1,4,"0.0"),("val_loss","验证损失",1,4,"0.0"),
            ("val_top1","验证 Top-1 (%)",100,100,"0"),("val_macro_f1","验证 Macro-F1 (%)",100,100,"0"),
            ("lr","学习率",1,.0011,"0.0000")]
    for i,(key,title,mul,hi,fmt) in enumerate(keys):
        x,y=.6+(i%3)*4.16,1.72+(i//3)*2.43
        chart(s,title,[str(r["epoch"]) for r in baseline],
              [("Baseline",[float(r[key])*mul for r in baseline]),("opt_combo",[float(r[key])*mul for r in opt])],
              x,y,3.98,2.2,lo=0,hi=hi,fmt=fmt)
    bt, ot = data("outputs/metrics/baseline_test.json"), data("outputs/metrics/opt_combo_test.json")
    text(s,"测试集结果（3669 张）",8.93,4.15,3.8,.35,16,CYAN,True)
    table(s,[["指标","Baseline","组合"],["Top-1",f"{bt['top1']*100:.2f}%",f"{ot['top1']*100:.2f}%"],
             ["Top-5",f"{bt['top5']*100:.2f}%",f"{ot['top5']*100:.2f}%"],
             ["Macro-F1",f"{bt['macro_f1']*100:.2f}%",f"{ot['macro_f1']*100:.2f}%"]],
          8.93,4.59,3.8,1.35,font=11)
    text(s,"前几轮提升最快，后段逐渐稳定。\n软标签改变 train loss，不能直接横比。\n单种子结果不足以判断稳定提升。",8.93,6.1,3.8,.78,12,MUTED)

    s=prs.slides[9]
    clear_slide(s,"错在哪里：多数是同物种内的品种混淆",
                "Baseline · Pet-37 test · 3669 张 · 行归一化混淆矩阵；右侧保留失败案例",
                10,"来源：outputs/confusion_matrix/baseline_cm.png、baseline_cat_dog_block.json；outputs/predictions/case_wrong_baseline_case*.png")
    picture(s,"outputs/confusion_matrix/baseline_cm.png",.6,1.7,6.5,4.85)
    cb=data("outputs/confusion_matrix/baseline_cat_dog_block.json")
    text(s,f"{cb['n_error']} 个错误中，{cb['n_cross_species_error']} 个跨猫狗物种\n其余 {cb['n_within_species_error']} 个是同物种品种判断错误",7.3,1.73,5.3,.8,20,CYAN,True)
    picture(s,"outputs/predictions/case_wrong_baseline_case01.png",7.3,2.75,2.5,2.5)
    picture(s,"outputs/predictions/case_wrong_baseline_case02.png",10,2.75,2.65,2.5)
    text(s,"这些失败图提示：相似外形、遮挡和背景\n都可能影响判断。不能仅凭热力图确定因果。",7.3,5.55,5.2,.8,17)
    text(s,"下一步：补充容易混淆的品种和不同拍摄角度。",7.3,6.48,5.2,.35,14,GREEN)

    s=prs.slides[10]
    clear_slide(s,"Grad-CAM：同时看判对和判错的图片",
                "Baseline · 挂载 stages[-1].blocks[-1] · 原图 / 热力图 / 叠加图；展示 2 个正确 + 2 个失败",
                11,"来源：outputs/gradcam/gradcam_{correct,wrong}_baseline_*.png；外部图片完整 Top-5 见 outputs/predictions/external_top5_pet37_grid5.png",
                "python tools/gradcam.py --help\n挂载点与原始图生成配置见 outputs/gradcam/ 下的 JSON。\n外部实拍样本来自 ImageNet 跨集合图片，8/8 判对只是小样本观察，不能证明跨域泛化或没有过拟合。")
    cams=[("正确：Bengal","gradcam_correct_baseline_Bengal_30_correct.png"),
          ("正确：Russian Blue","gradcam_correct_baseline_Russian_Blue_205_correct.png"),
          ("失败：Boxer","gradcam_wrong_baseline_boxer_2_wrong.png"),
          ("失败：Egyptian Mau","gradcam_wrong_baseline_Egyptian_Mau_204_wrong.png")]
    for i,(title,fn) in enumerate(cams):
        x,y=.65+(i%2)*6.35,1.73+(i//2)*2.1
        text(s,title,x,y,5.8,.3,15,GREEN if i<2 else ORANGE,True)
        picture(s,"outputs/gradcam/"+fn,x,y+.38,5.95,1.52)
    text(s,"关注到主体 ≠ 品种一定判断正确。热力图用于检查线索，仍要结合失败案例和完整测试集。",.65,6.28,12,.55,18)

    rep=data("outputs/reparam/repvit_m0_9_pet37_reparam_report.json")
    s=prs.slides[11]
    clear_slide(s,"重参数化：合并分支后，误差仍在设定阈值内",
                "Pet-37 / baseline_best.pt · CPU FP32 · eval → 深拷贝 → fuse · 固定随机输入 n=32，batch=8，224×224，seed=20240912",
                12,"来源：outputs/reparam/repvit_m0_9_pet37_reparam_report.json；与第 13 页真实图片 ONNX 实验分开",
                "python tools/reparam_verify.py --model repvit_m0_9_pet37 --weights checkpoints/baseline_best.pt --num-samples 32 --batch-size 8 --seed 20240912 --skip-onnx --out-dir outputs/verification/reparam_pet37\n"
                "max_abs_err=7.092952728271484e-06；mean_abs_err=2.030726818702533e-06。32 个输入是 torch.randn，不是真实测试图片。\n"
                "权重 missing=0 / unexpected=0。BN 模块 107→0；ONNX 图的 BN 节点 24→0 是另一种统计，不可混用。\n"
                "历史官方 C=1000 报告存在权重键不匹配，不纳入结论；见 report/LOGITS_AUDIT.md。")
    chart(s,"PyTorch 模块数量",[ "BatchNorm","Conv2d"],
          [("融合前",[rep["before"]["n_bn"],rep["before"]["n_conv2d"]]),
           ("融合后",[rep["after"]["n_bn"],rep["after"]["n_conv2d"]])],
          .65,1.8,6,3.75,XL_CHART_TYPE.COLUMN_CLUSTERED,lo=0,hi=140,fmt="0")
    table(s,[["数值验证（随机输入）","结果"],["最大 |Δlogits|",f"{rep['diff']['max_abs_err']:.3e}"],
             ["平均 |Δlogits|",f"{rep['diff']['mean_abs_err']:.3e}"],["Top-1 一致","32 / 32"],
             ["Top-5 最小交集","5 / 5"],["阈值 / 判定","max < 1e-4 / PASS"]],
          7,1.95,5.6,3.3,sizes=[3.3,2.3],font=17)
    text(s,"先把 BN 的固定统计量折进卷积，再把 1×1、3×3 与 identity 分支的核相加。",.65,5.95,12,.6,20)
    text(s,"代数上等价；FP32 运算顺序改变会产生舍入误差。这里通过的是已测输入和设定阈值。",.65,6.58,12,.35,14,MUTED)

    s=prs.slides[12]
    clear_slide(s,"PyTorch ↔ ONNX：固定 n=12，比较同一输入张量",
                "融合后的 PyTorch vs ORT CPUExecutionProvider · FP32 · batch=1 · 224×224 · 每个模型各 12 张",
                13,"来源：outputs/metrics/consistency_{repvit_m0_9_pet37,repvit_m0_9_in1k,repvit_m1_0_in1k}.json",
                "python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --images datasets/lists/pet_test.txt --limit 12 --out outputs/verification/consistency_repvit_m0_9_pet37_n12.json\n"
                "Pet-37 顺序取固定列表前 12 张，每张预处理一次，同一 numpy 张量送入两端。\n"
                "max=6.198883056640625e-06；mean=1.5006899711048998e-06；Top-1/Top-5 集合均 1.0。\n"
                "5.25e-06 是未找到对应完整产物的旧引用；旧命令写 limit=8 不能证明该值来自 n=8。当前权重跑 n=8 也不能复现旧值。\n"
                "不是全测试集一致率，不验证两套预处理独立实现等价，也不能排除所有部署错误。")
    cells=[["模型 / 输入列表","n","最大 |Δlogits|","平均 |Δlogits|","Top-1 / Top-5"]]
    for key,label in [("repvit_m0_9_pet37","M0.9 Pet-37 / pet_test"),
                      ("repvit_m0_9_in1k","M0.9 / ImageNet 子集"),("repvit_m1_0_in1k","M1.0 / ImageNet 子集")]:
        d=data("outputs/metrics/consistency_"+key+".json")
        cells.append([label,d["n"],f"{d['max_abs_logits']:.3e}",f"{d['mean_abs_logits']:.3e}","100% / 100%"])
    table(s,cells,.65,1.9,12,2.15,sizes=[3.65,.6,2.4,2.4,2.95],font=16)
    text(s,"1  部署预处理由 PIL 实现；本次对比把同一张量送入两个推理后端。\n2  结论仅覆盖这 12 张图片，不能写成完整测试集或全部六模型一致。\n3  旧的 5.25e-06 缺少对应完整记录，当前统一引用上表落盘值。",.7,4.45,12,1.5,20)
    text(s,"复跑：python deploy/compare_torch_onnx.py --model repvit_m0_9_pet37 --limit 12\n完整命令、固定列表与输出路径已写入本页备注和 report/LOGITS_AUDIT.md。",.7,6.12,12,.73,15,GREEN)

    s=prs.slides[13]
    clear_slide(s,"部署怎么选：先看同一机器上的延迟和准确率",
                "i7-13650HX · Windows 11 · ORT 1.30.0 CPUExecutionProvider · FP32 · 1×3×224×224 · 预热10 + 正式50 · threads=4",
                14,"来源：outputs/benchmarks/summary.csv；outputs/pretrained_eval/<model>/metrics.json；各模型完整元信息见 benchmark JSON",
                "python deploy/benchmark.py --model repvit_m0_9_in1k --model repvit_m1_0_in1k --model repvit_m0_9_pet37 --warmup 10 --runs 50 --threads 4 --out-dir outputs/verification/benchmarks\n"
                "左图含六个部署模型，仅比较速度；右图只放相同 ImageNet 子集的五个型号，不把 Pet-37 准确率混进来。\n"
                "官方 iPhone 延迟与本机 CPU 延迟不可直接比较。性能数据是历史落盘值，现场新测会有波动。")
    br=rows("outputs/benchmarks/summary.csv")
    chart(s,"六模型推理耗时 (ms)",["M0.9","M1.0","M1.1","M1.5","M2.3","Pet37"],
          [("P50",[float(r["p50_ms"]) for r in br]),("P95",[float(r["p95_ms"]) for r in br])],
          .65,1.85,6.05,4.25,XL_CHART_TYPE.COLUMN_CLUSTERED,lo=0,hi=36,fmt="0")
    text(s,"同一 ImageNet 子集：速度与准确率",7.05,1.85,5.6,.3,16,CYAN,True)
    xy=XyChartData()
    for r in br[:5]:
        arch=r["model"].replace("_in1k","")
        d=data("outputs/pretrained_eval/"+arch+"/metrics.json")
        # Metrics store Top-1 as percentage.
        acc=d.get("top1",d.get("accuracy_top1"))
        if acc is None:
            raise KeyError(f"missing top1: {arch}")
        se=xy.add_series(arch.replace("repvit_","").replace("_","."))
        se.add_data_point(float(r["p50_ms"]),float(acc))
    bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(7.05), Inches(2.2), Inches(5.6), Inches(3.9))
    bg.fill.solid(); bg.fill.fore_color.rgb=RGBColor.from_string("FFFFFF"); bg.line.fill.background()
    ch=s.shapes.add_chart(XL_CHART_TYPE.XY_SCATTER, Inches(7.05), Inches(2.2), Inches(5.6), Inches(3.9), xy).chart
    ch.has_legend=True; ch.legend.position=XL_LEGEND_POSITION.BOTTOM; ch.legend.font.size=Pt(11)
    ch.font.size=Pt(11)
    ch.category_axis.has_title=True; ch.category_axis.axis_title.text_frame.text="P50 (ms)"
    ch.value_axis.has_title=True; ch.value_axis.axis_title.text_frame.text="Top-1 (%)"
    ch.value_axis.minimum_scale=77; ch.value_axis.maximum_scale=84
    for se in ch.series:
        se.marker.style=XL_MARKER_STYLE.CIRCLE; se.marker.size=10
    text(s,"本子集上 M1.0 / M1.1 的 Top-1 相同，M1.0 更快。更大模型的额外耗时，需要结合使用场景判断。",.65,6.35,12,.62,18)
    prs.save(PPT)
    # The brief is regenerated from actual slide content and notes, preventing source drift.
    out=["# 答辩 PPT 内容与证据", "", "15 页；图表由 tools/update_defense.py 从落盘 CSV/JSON 填充。", ""]
    for i,s in enumerate(prs.slides,1):
        out += [f"## P{i:02d}", ""]
        def values(sh):
            if getattr(sh,"has_text_frame",False):
                yield sh.text
            if getattr(sh,"has_table",False):
                for row in sh.table.rows:
                    yield " / ".join(c.text for c in row.cells)
            if hasattr(sh,"shapes"):
                for c in sh.shapes:
                    yield from values(c)
        for sh in s.shapes:
            out.extend(v.replace("\v","\n") for v in values(sh) if v.strip())
        if s.has_notes_slide:
            out += ["", "备注与验证命令：", s.notes_slide.notes_text_frame.text]
        out += [""]
    (ROOT/"report/PPT_CONTENT.md").write_text("\n\n".join(out),encoding="utf-8")
    print("Updated 15 slides; charts from saved CSV/JSON; notes and PPT_CONTENT synchronized.")


if __name__ == "__main__":
    main()
