# tools/run_ablation.py —— 依次跑 baseline / A / B / A+B，写统一的 ablation_summary.csv
import argparse, os, subprocess, sys, json, time
import pandas as pd, yaml

ORDER = [("baseline", "configs/ablation/baseline.yaml"),
         ("expA",     "configs/ablation/exp_a.yaml"),
         ("expB",     "configs/ablation/exp_b.yaml"),
         ("expAB",    "configs/ablation/exp_ab.yaml")]

def read_run(name, cfg, log, wall_clock):
    """从规范产物里取一组结果：逐 epoch 日志 + test 指标（路径由 experiment_name 决定）。"""
    csv_path = f"outputs/logs/{name}_metrics.csv"
    assert os.path.exists(csv_path), f"{name} 未产出 {csv_path}"
    df = pd.read_csv(csv_path)          # 表头见模块 6 §6.4.1，val_* 三列是 0~1 小数
    best_row = df.loc[df["val_macro_f1"].idxmax()]     # best 由 eval.metric_for_best 决定
    m = dict(experiment_name=name, best_epoch=int(best_row["epoch"]),
             epochs=int(df["epoch"].max()), n_iters=int(len(df)) * 45,   # 45 = 2940 // 64
             # 统一乘 100 变成百分数，与下面交互项判据里的「0.5 个点」阈值同口径
             top1=100.0 * float(best_row["val_top1"]),
             top5=100.0 * float(best_row["val_top5"]),
             macro_f1=100.0 * float(best_row["val_macro_f1"]))
    test_path = f"outputs/metrics/{name}_test.json"    # test 只跑一次，指标同样是 0~1 小数
    if os.path.exists(test_path):
        with open(test_path, encoding="utf-8") as f:
            t = json.load(f)
        m["test_top1"] = 100.0 * float(t["top1"])
        m["test_macro_f1"] = 100.0 * float(t["macro_f1"])
    with open(cfg, encoding="utf-8") as f:
        c = yaml.safe_load(f)
    m.update({k: c["train"][k] for k in
              ("mixup_alpha", "cutmix_alpha", "mixup_prob", "mixup_switch_prob",
               "label_smoothing", "epochs") if k in c["train"]})
    m.update(warmup_epochs=c["optim"]["warmup_epochs"], lr=c["optim"]["lr"],
             weight_decay=c["optim"]["weight_decay"], optimizer=c["optim"]["optimizer"],
             batch_size=c["data"]["batch_size"], seed=c["seed"])
    m["wall_clock_sec"] = wall_clock
    m["log"] = log
    return m

def run_one(name, cfg, outdir, extra=()):
    """调唯一的训练入口 tools/train.py（配置一律用 --cfg，覆盖一律用 --set），日志同时落盘。

    四组实验靠 config 里的 `experiment_name` 区分，产物一律写成
    `outputs/logs/<experiment_name>_metrics.csv`、`checkpoints/<experiment_name>_best.pt`
    与 `outputs/metrics/<experiment_name>_test.json`，因此四组串行天然互不覆盖，
    不需要额外的输出目录参数（全仓只认 `--cfg` 与 `--set` 两个配置相关开关）。
    """
    log = os.path.join(outdir, f"train_{name}.log")
    cmd = [sys.executable, "tools/train.py", "--cfg", cfg, *extra]
    print("RUN:", " ".join(cmd))
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    assert p.returncode == 0, f"{name} 训练失败，见 {log}"
    return read_run(name, cfg, log, round(time.time() - t0, 1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/ablation")
    ap.add_argument("--only", nargs="*", default=[n for n, _ in ORDER])
    ap.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                    help="透传给 train.py 的额外参数，必须写成 --extra --set xxx=yyy 的形式")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    rows = [run_one(n, c, a.out_dir, a.extra) for n, c in ORDER if n in a.only]
    df = pd.DataFrame(rows)
    cols = ["experiment_name", "top1", "top5", "macro_f1", "best_epoch", "epochs",
            "n_iters", "wall_clock_sec", "params_M", "test_top1", "test_macro_f1",
            "mixup_alpha", "cutmix_alpha", "mixup_prob", "mixup_switch_prob",
            "label_smoothing", "warmup_epochs", "optimizer", "lr", "weight_decay",
            "batch_size", "seed", "log"]
    df[[c for c in cols if c in df.columns]].to_csv(
        os.path.join(a.out_dir, "ablation_summary.csv"), index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
