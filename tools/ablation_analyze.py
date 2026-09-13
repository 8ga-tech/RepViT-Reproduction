# tools/ablation_analyze.py —— 交互项 + 叠加/冲突判定 JSON
import argparse, json, os
import numpy as np, pandas as pd

def interaction(df, metric, names=("baseline", "expA", "expB", "expAB")):
    """交互项 = dAB - (dA + dB)。组名可配置——规格书 11.2.2 用的是 expA/expB/expAB，
    本仓实验标识符遵循 §4.3 的 <experiment_name> 约定（opt_abl_a / opt_abl_b / opt_abl_ab），
    两者都合法，因此把组名做成参数而不是写死，避免为了迁就工具而改实验命名。"""
    g = df.set_index("experiment_name")[metric]
    b, nA, nB, nAB = names
    dA = g[nA] - g[b]; dB = g[nB] - g[b]
    dAB = g[nAB] - g[b]
    return dict(metric=metric, delta_A=float(dA), delta_B=float(dB), delta_AB=float(dAB),
                interaction=float(dAB - (dA + dB)),
                predicted_additive=float(g[b] + dA + dB),
                observed_AB=float(g[nAB]),
                group_names=dict(baseline=b, A=nA, B=nB, AB=nAB),
                verdict=("super-additive(协同)" if dAB - (dA + dB) > 0.5 else
                         "sub-additive(次加性/冗余)" if dAB - (dA + dB) < -0.5 else "additive(可加)"))

def noise_floor(df, outdir):
    """实用判据：增益是否超过随机波动。
    单种子时的保守估计——用同一配置内两个不同 best epoch 的验证精度差、
    或同类方法已知的种子噪声（Pet 37 类、740 张验证集，Top-1 的 ±1σ 约 0.6~0.9 点）。
    严格做法见 --seeds。"""
    v = df.set_index("experiment_name")["top1"]
    spread = float(v.max() - v.min())
    return dict(single_seed_estimated_sigma_top1=0.8,
                rule="|delta| > 2*sigma 才判定为真实增益；否则只能写『在随机波动范围内』",
                observed_max_spread=spread, n_val=740,
                binom_ci_half_width_95=round(1.96 * float(np.sqrt(0.9*0.1/740))*100, 2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/ablation/ablation_summary.csv")
    ap.add_argument("--out", default="outputs/advanced/ablation_summary.json")
    ap.add_argument("--name-a", default="expA", help="实验 A 的 experiment_name")
    ap.add_argument("--name-b", default="expB", help="实验 B 的 experiment_name")
    ap.add_argument("--name-ab", default="expAB", help="A+B 组合的 experiment_name")
    a = ap.parse_args()
    json_out = a.out.replace(".csv", ".json")
    if not os.path.exists(a.csv):
        raise SystemExit(f"缺少 {a.csv}；请先用 tools/run_ablation.py 生成，"
                         f"或直接把汇总表写到该路径")
    df = pd.read_csv(a.csv)
    names = ("baseline", a.name_a, a.name_b, a.name_ab)
    need = set(names)
    missing = need - set(df["experiment_name"])
    if missing: raise SystemExit(f"缺少实验组: {missing}，请先跑 run_ablation.py")
    res = dict(interactions=[interaction(df, m, names) for m in ("top1", "macro_f1", "top5")],
               noise=noise_floor(df, os.path.dirname(a.out)),
               table=df.to_dict("records"))
    os.makedirs(os.path.dirname(json_out) or ".", exist_ok=True)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=True, indent=2, default=str)
    for r in res["interactions"]:
        print(f"{r['metric']:10s} dA={r['delta_A']:+.3f} dB={r['delta_B']:+.3f} "
              f"dAB={r['delta_AB']:+.3f} I={r['interaction']:+.3f} -> {r['verdict']}")
