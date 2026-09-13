# tools/pareto_family.py —— 帕累托前沿 + 边际收益 + 推荐型号结论表
import argparse, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

def pareto_front(df, cost_col, acc_col="top1"):
    """代价越小越好、精度越大越好。返回未被支配的行（前端），按代价升序。"""
    keep = []
    for i, r in df.iterrows():
        dom = ((df[cost_col] <= r[cost_col]) & (df[acc_col] >= r[acc_col]) &
               ((df[cost_col] < r[cost_col]) | (df[acc_col] > r[acc_col])))
        if not dom.any(): keep.append(i)
    return df.loc[keep].sort_values(cost_col)

def plot(df, cost_col, xlabel, out_png, title):
    front = pareto_front(df, cost_col)
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    ax.scatter(df[cost_col], df["top1"], s=48, c="#7f8c8d", zorder=3, label="all variants")
    ax.plot(front[cost_col], front["top1"], "-o", color="#c0392b", lw=1.6, ms=7,
            zorder=4, label="Pareto front")
    for _, r in df.iterrows():
        ax.annotate(r["model"].replace("repvit_", "").replace("_in1k", ""),
                    (r[cost_col], r["top1"]), textcoords="offset points",
                    xytext=(6, -10), fontsize=8)
    ax.set_xlabel(xlabel); ax.set_ylabel("ImageNet-1K subset Top-1 (%)")
    ax.set_title(title); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)
    print("saved", out_png)
    return front

def marginal(df, lat_col="lat_p50_ms"):
    """边际收益：按延迟升序，算 d(acc)/d(lat) 与 d(acc)/d(params)"""
    d = df.sort_values(lat_col).copy()
    d["acc_per_ms"] = d["top1"] / d[lat_col]
    d["d_top1"] = d["top1"].diff()
    d["d_lat"] = d[lat_col].diff()
    d["d_params"] = d["params_M"].diff()
    d["marginal_acc_per_ms"] = d["d_top1"] / d["d_lat"]
    d["marginal_acc_per_M"] = d["d_top1"] / d["d_params"]
    return d

def recommend(df, lat_budget_ms=15.0, device="Intel Iris Xe iGPU / 4-thread CPU"):
    """在延迟预算内取精度最高者；预算外全部剔除。输出结论表（约束/得分/理由）。
    得分 = 预算内归一化精度 0.7 + 归一化 P95 稳定性 0.3，权重可调但要写进报告。"""
    cand = df[df["lat_p50_ms"] <= lat_budget_ms].copy()
    relaxed = False
    if cand.empty:
        # 预算内一个都放不下时，退回「取最快的那个」，并在理由里**明说这是放宽后的结论**。
        # 原稿只换了数据却沿用「落在预算内」的措辞，会输出
        # 「P50=36.08ms 落在 15.0ms 预算内」这种自相矛盾的结论（实测复现）。
        cand = df.nsmallest(1, "lat_p50_ms").copy()
        relaxed = True
    cand["score"] = 0.7 * (cand["top1"] / cand["top1"].max()) + \
                    0.3 * (cand["lat_p95_ms"].min() / cand["lat_p95_ms"])
    cand = cand.sort_values("score", ascending=False)
    out = []
    for _, r in cand.iterrows():
        if relaxed:
            reason = (f"【放宽结论】没有任何型号的 P50 <= {lat_budget_ms}ms 预算，"
                      f"因此取全家族中延迟最低者：P50={r['lat_p50_ms']:.2f}ms"
                      f"（超出预算 {r['lat_p50_ms']-lat_budget_ms:.2f}ms）；"
                      f"若要满足预算需换更快硬件或对模型做量化/裁剪")
        else:
            reason = (f"P50={r['lat_p50_ms']:.2f}ms 落在 {lat_budget_ms}ms 预算内；"
                      f"Top-1 相对该预算内最优 {cand['top1'].max():.2f}% 差 "
                      f"{cand['top1'].max()-r['top1']:.2f} 个点")
        out.append(dict(device=device, constraint=f"P50 <= {lat_budget_ms} ms",
                        model=r["model"], score=round(float(r["score"]), 4),
                        top1=r["top1"], lat_p50_ms=r["lat_p50_ms"],
                        lat_p95_ms=r["lat_p95_ms"], reason=reason))
    return pd.DataFrame(out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/benchmarks/family_summary.csv")
    ap.add_argument("--out-dir", default="outputs/advanced")
    ap.add_argument("--lat-budget-ms", type=float, default=15.0)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    df = pd.read_csv(a.csv)

    # 延迟口径：Pareto 要用**部署口径**（ONNX Runtime CPU）而不是 PyTorch 前向。
    # family_summary.csv 里的 lat_* 是 PyTorch CPU 计时（36~101ms），
    # 而实际交付的是 ONNX（7~33ms），两者相差数倍、不可混用。
    # 这里从 outputs/benchmarks/<key>_benchmark.json 读 ONNX 实测 P50/P95 覆盖，
    # 并把来源写进 lat_source 列，避免报告里出现「看起来同源、实际不同源」的数字。
    import glob as _glob
    onnx_lat = {}
    for fp in _glob.glob("outputs/benchmarks/*_benchmark.json"):
        try:
            _j = json.load(open(fp, encoding="utf-8"))
            _k = _j.get("model") or os.path.basename(fp).replace("_benchmark.json", "")
            if "p50_ms" in _j:
                onnx_lat[_k] = (_j["p50_ms"], _j.get("p95_ms", _j["p50_ms"]))
        except Exception:
            pass
    if onnx_lat:
        df["lat_pt_p50_ms"] = df["lat_p50_ms"]
        df["lat_source"] = "pytorch_cpu_threads" + str(a.__dict__.get("threads", ""))
        for i, row in df.iterrows():
            m = row["model"]
            if m in onnx_lat:
                df.at[i, "lat_p50_ms"] = onnx_lat[m][0]
                df.at[i, "lat_p95_ms"] = onnx_lat[m][1]
                df.loc[i, "lat_source"] = "onnxruntime_CPUExecutionProvider"
        print(f"[lat] 已用 ONNX 实测延迟覆盖 {sum(1 for m in df['model'] if m in onnx_lat)}/{len(df)} 个型号")

    f_p = plot(df, "params_M", "Params (M)", f"{a.out_dir}/params_vs_acc.png", "Params vs. Accuracy")
    f_l = plot(df, "lat_p50_ms", "Latency P50 (ms, ORT CPU, bs=1, FP32)",
               f"{a.out_dir}/latency_vs_acc.png", "Latency vs. Accuracy")
    f_m = plot(df, "macs_G", "MACs (G)", f"{a.out_dir}/macs_vs_acc.png", "MACs vs. Accuracy")
    d = marginal(df)
    print(d[["model", "lat_p50_ms", "params_M", "macs_G", "top1",
             "acc_per_ms", "marginal_acc_per_ms", "marginal_acc_per_M"]].to_string(index=False))
    d.to_csv(f"{a.out_dir}/marginal_returns.csv", index=False, encoding="utf-8-sig")
    rec = recommend(df, a.lat_budget_ms)
    rec.to_csv(f"{a.out_dir}/model_recommendation.csv", index=False, encoding="utf-8-sig")
    with open(f"{a.out_dir}/pareto_summary.json", "w", encoding="utf-8") as fh:
        json.dump(dict(pareto_by_params=f_p["model"].tolist(),
                       pareto_by_latency=f_l["model"].tolist(),
                       pareto_by_macs=f_m["model"].tolist(),
                       recommended=rec.to_dict("records")), fh, ensure_ascii=True, indent=2)
    print(rec.to_string(index=False))
