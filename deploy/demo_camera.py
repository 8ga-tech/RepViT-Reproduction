# deploy/demo_camera.py —— 实时 Top-5 + 端到端 FPS + 按键切换型号 + 抖动分析
#
# 产物口径（X-6）：本机没有摄像头、也没有视频素材，因此**仓库里没有
# outputs/benchmarks/realtime_jitter.json**——本脚本默认不落盘，不声明一个找不到的产物。
# 现场跑通后按下面这条命令留证据：
#   python deploy/demo_camera.py --source 0 --model 1 \
#       --jitter-report outputs/benchmarks/realtime_jitter.json
import argparse, json, time
from collections import deque
from pathlib import Path
import cv2, numpy as np, onnxruntime as ort
try:
    from deploy.model_registry import onnx_path, labels_for   # 禁止硬写 ONNX 文件名
except ImportError:      # `python deploy/X.py` 时 sys.path[0]=deploy/
    from model_registry import onnx_path, labels_for

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD  = np.array([0.229, 0.224, 0.225], np.float32)
MODELS = {  # 现场按 1/2/3 切换（值都是 registry 键名）；session 必须缓存，否则切换要卡几秒
    "1": "repvit_m0_9_in1k",
    "2": "repvit_m1_0_in1k",
    "3": "repvit_m0_9_pet37",
}

def preprocess(frame_bgr, size=224, crop_pct=0.95):
    """必须与训练/离线评测完全一致：BGR→RGB、按 crop_pct=0.95 缩短边、center crop、
    /255、mean/std、NCHW。写成 Resize(256) 会与模型训练时的有效视野不一致。"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    s = size / crop_pct / min(h, w)
    img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_LINEAR)
    nh, nw = img.shape[:2]
    top, left = (nh - size) // 2, (nw - size) // 2
    img = img[top:top + size, left:left + size]
    x = img.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])

_cache = {}
def load(key):
    if key not in _cache:
        sess = ort.InferenceSession(onnx_path(key), providers=["CPUExecutionProvider"])
        _cache[key] = (sess, labels_for(key))     # labels_for 已返回去空白的标签列表
    return _cache[key]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=摄像头；也可给视频文件路径")
    ap.add_argument("--model", default="1", choices=list(MODELS))
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--jitter-report", default=None,
                    help="把抖动/置信度统计写到该路径；**默认不落盘**（仓库未提交 "
                         "realtime_jitter.json，本机无摄像头也无视频素材）。"
                         "现场留证据：--jitter-report outputs/benchmarks/realtime_jitter.json")
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    cap = (cv2.VideoCapture(src, cv2.CAP_DSHOW) if isinstance(src, int)
           else cv2.VideoCapture(src))     # Windows 上显式 CAP_DSHOW 可显著加快打开
    if not cap.isOpened(): raise SystemExit("无法打开视频源")
    sess, names = load(a.model); iname = sess.get_inputs()[0].name
    t_fps, conf_hist, pred_hist = deque(maxlen=30), deque(maxlen=60), deque(maxlen=60)
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok: break
        logits = sess.run(None, {iname: preprocess(frame)})[0][0]
        p = np.exp(logits - logits.max()); p /= p.sum()
        top = np.argsort(-p)[:a.k]
        conf_hist.append(float(p[top[0]])); pred_hist.append(int(top[0]))
        for i, idx in enumerate(top):
            cv2.putText(frame, f"{i+1}. {names[idx][:28]} {p[idx]*100:5.1f}%",
                        (12, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 0) if i == 0 else (200, 200, 200), 2)
        cv2.putText(frame, f"FPS(avg30)={np.mean(t_fps):5.1f}  model={a.model}  "
                           f"EP={sess.get_providers()[0]}",
                    (12, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
        cv2.imshow("RepViT realtime", frame)
        k = cv2.waitKey(1) & 0xFF
        # 端到端计时点必须在绘制与显示之后：设计要点要求 FPS 覆盖解码+预处理+推理+绘制四段，
        # 记在 sess.run 之后会把 putText/imshow/waitKey 的开销全部漏掉，FPS 系统性偏高
        t_fps.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        if k == 27: break
        if chr(k) in MODELS and chr(k) != a.model:          # 现场切换型号
            a.model = chr(k); sess, names = load(a.model); iname = sess.get_inputs()[0].name
            conf_hist.clear(); pred_hist.clear()
    cap.release(); cv2.destroyAllWindows()
    flips = sum(1 for i in range(1, len(pred_hist)) if pred_hist[i] != pred_hist[i - 1])
    rep = dict(model=a.model, frames=len(pred_hist),
               flicker_rate=flips / max(len(pred_hist) - 1, 1),
               conf_std=float(np.std(conf_hist)), conf_mean=float(np.mean(conf_hist)),
               fps_avg30=float(np.mean(t_fps)),
               note="FPS 覆盖解码+预处理+推理+绘制四段（计时点在 waitKey 之后），只统计 sess.run 会虚高")
    if a.jitter_report:
        out = Path(a.jitter_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"[write] {a.jitter_report}")
    print(json.dumps(rep, ensure_ascii=True))

if __name__ == "__main__":
    main()
