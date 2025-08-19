# analyze_efficiency_spp_gamma_depth.py
import os, sys, torch, yaml, numpy as np, matplotlib.pyplot as plt, optuna
from optuna.samplers import NSGAIISampler
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo import Model
from score import compute_zico_score_avg

device = torch.device('cpu')
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---- 探索空間 ----
YAML_PATH = "models/yolov5n.yaml"
DEPTH_WIDTH_MAP = {0.33:0.25, 0.76:0.25}
DEPTH_CAND = sorted(DEPTH_WIDTH_MAP.keys())
GAMMA_CAND = [1.0, 1.25, 1.5]
SPP_MODE_CAND = ["none", "sppf", "spp"]

# 固定パラメータ
E_FIXED   = 1.0
ACT_FIXED = "SiLU"

# results: (zico, params, lat_us, depth, gamma, mode, flops)
results = []

# ---------- cfg helpers ----------
def dcopy(x): return yaml.safe_load(yaml.dump(x))

def apply_depth_width(cfg, d, w):
    cfg = dcopy(cfg)
    for layer in cfg['backbone']:
        if layer[2] == 'C3':
            layer[1] = max(1, int(round(layer[1] * d)))
    for layer in cfg['backbone']:
        if isinstance(layer[3], dict) and 'c2' in layer[3]:
            layer[3]['c2'] = max(1, int(round(layer[3]['c2'] * w)))
    return cfg

def set_expansion(cfg, e_scale=1.0):
    cfg = dcopy(cfg)
    for section in ('backbone','head'):
        for layer in cfg[section]:
            args = layer[3]
            if isinstance(args, dict) and 'e' in args:
                try: args['e'] = float(args['e']) * float(e_scale)
                except: pass
    return cfg

def set_activation(cfg, act="SiLU"):
    cfg = dcopy(cfg)
    cfg["activation"] = act
    for section in ('backbone','head'):
        for layer in cfg[section]:
            args = layer[3]
            if isinstance(args, dict):
                args["act"] = act
    return cfg

# --- SPP/SPPF/なしをcfgで切り替え（dict/list両対応） ---
def set_spp_mode_in_cfg(cfg, mode="sppf"):
    assert mode in ("sppf","spp","none")
    cfg = dcopy(cfg)

    def get_c2(args):
        if isinstance(args, dict): return int(args.get("c2")) if "c2" in args else None
        if isinstance(args, (list,tuple)) and len(args)>=1: return int(args[0])
        return None

    def set_c2(args, c2):
        if c2 is None: return args
        if isinstance(args, dict):
            args["c2"] = int(c2)
        elif isinstance(args, list):
            if len(args)==0: args.append(int(c2))
            else: args[0] = int(c2)
        else:
            args = [int(c2)]
        return args

    def set_k_sppf(args, k=5):
        if isinstance(args, dict):
            args["k"] = int(k)
        elif isinstance(args, list):
            while len(args)<2: args.append(None)
            args[1] = int(k)
        return args

    def set_k_spp(args, ks=(5,9,13)):
        ks = list(ks)
        if isinstance(args, dict):
            args["k"] = ks
        elif isinstance(args, list):
            while len(args)<2: args.append(None)
            args[1] = ks
        return args

    def make_conv1x1_args(args, c2):
        if isinstance(args, dict):
            args["c2"] = int(c2) if c2 is not None else int(args.get("c2", 0))
            args["k"] = 1; args["s"] = 1; args["act"] = ACT_FIXED
            return args
        elif isinstance(args, list):
            c2v = int(c2) if c2 is not None else (args[0] if len(args)>=1 else 0)
            return [c2v, 1, 1]
        return args

    for sec in ("backbone","head"):
        for l in cfg.get(sec, []):
            if not (isinstance(l, list) and len(l)>=4): continue
            mtype, args = l[2], l[3]
            if mtype not in ("SPPF","SPP"): continue
            c2 = get_c2(args)
            if mode=="sppf":
                l[2]="SPPF"; l[3]=set_k_sppf(args, k=5)
            elif mode=="spp":
                l[2]="SPP";  l[3]=set_k_spp(set_c2(args, c2), ks=(5,9,13))
            else:  # none
                l[2]="Conv"; l[3]=make_conv1x1_args(args, c2)
    return cfg

# ---------- metrics ----------
def compute_model_metrics(model, dummy):
    zico = compute_zico_score_avg(model, dummy, runs=3, seed=42)
    params = sum(p.numel() for p in model.parameters())
    # CPU profile（粗い遅延プロキシ）
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        with torch.profiler.record_function("model_inference"):
            _ = model(dummy)
    lat_us = sum(e.cpu_time_total for e in prof.key_averages() if e.cpu_time_total is not None)
    flops = None
    try:
        from thop import profile
        model.eval()
        with torch.no_grad():
            macs, _ = profile(model, inputs=(dummy,), verbose=False)
        flops = 2 * macs
    except Exception:
        pass
    return float(zico), int(params), float(lat_us), (None if flops is None else float(flops))

# ---------- Optuna objective（多目的: ZiCo↑, Params↓, FLOPs↓） ----------
def objective(trial):
    d     = trial.suggest_categorical("depth", DEPTH_CAND)
    gamma = trial.suggest_categorical("gamma", GAMMA_CAND)
    mode  = trial.suggest_categorical("spp_mode", SPP_MODE_CAND)
    w     = DEPTH_WIDTH_MAP[d]

    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)
    # 順序：depth/width → spp_mode → e/act
    cfg = apply_depth_width(cfg, d, w)
    cfg = set_spp_mode_in_cfg(cfg, mode=mode)
    cfg = set_expansion(cfg, E_FIXED)
    cfg = set_activation(cfg, ACT_FIXED)

    model = Model(cfg).to(device)
    # BN γ スケール
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) and m.weight is not None:
                m.weight.mul_(float(gamma))

    dummy = torch.randn(1, 3, 640, 640, device=device)
    zico, params, lat_us, flops = compute_model_metrics(model, dummy)
    results.append((zico, params, lat_us, d, gamma, mode, flops))

    # FLOPsが取れない場合は巨大値でペナルティ（最小化なので不利）
    flops_obj = (flops if flops is not None else 1e18)

    print(f"[Trial] d={d:.2f}, w={w:.2f}, mode={mode}, γ={gamma} "
          f"=> ZiCo={zico:.2f}, Params={params/1e6:.2f}M, Lat(us)={lat_us:.0f}, FLOPs(G)={(flops/1e9) if flops else None}")

    return zico, params, flops_obj

# ---------- Pareto抽出 & 可視化 ----------
def _is_dominated(a, b, dirs):
    better_or_equal = True
    strictly_better = False
    for k, d in enumerate(dirs):
        diff = d * (b[k] - a[k])
        if diff < 0: better_or_equal = False; break
        if diff > 0: strictly_better = True
    return better_or_equal and strictly_better

def compute_pareto_front(recs):
    z = np.array([r[0] for r in recs], float)           # ↑
    p = np.array([r[1] for r in recs], float)           # ↓
    f = np.array([r[6] for r in recs], float)           # ↓ (raw FLOPs)
    mask = ~np.isnan(z) & ~np.isnan(p) & ~np.isnan(f)
    objs = np.stack([z[mask], p[mask], f[mask]], 1)
    dirs = (+1, -1, -1)
    idxs = np.where(mask)[0]
    n = objs.shape[0]
    keep = np.ones(n, bool)
    for i in range(n):
        if not keep[i]: continue
        for j in range(n):
            if i==j: continue
            if _is_dominated(objs[i], objs[j], dirs):
                keep[i]=False; break
    return idxs[keep]

def plot_unified_with_pareto(recs, front_indices, x_metric="flops", title_suffix="pareto"):
    z = np.array([r[0] for r in recs], float)
    pM = np.array([r[1] for r in recs], float)/1e6
    fG = np.array([(r[6]/1e9) if r[6] is not None else np.nan for r in recs], float)
    d  = np.array([r[3] for r in recs], float)
    g  = np.array([r[4] for r in recs], float)
    m  = np.array([r[5] for r in recs], object)

    if x_metric=="flops":
        x = fG; xlab="FLOPs (G)"; base_mask=~np.isnan(x)
    else:
        x = pM; xlab="Params (M)"; base_mask=~np.isnan(z)

    x_all, y_all = x[base_mask], z[base_mask]
    size_all = 20.0*(1.0+pM[base_mask])

    modes = ["none","sppf","spp"]
    marks = {"none":"o","sppf":"^","spp":"s"}
    gammas = sorted(set(g.tolist()))
    cmap = plt.get_cmap("tab10")
    color = {gammas[i]: cmap(i%10) for i in range(len(gammas))}

    plt.figure(figsize=(8,6))
    plt.scatter(x_all, y_all, s=size_all, c="#bbbbbb", alpha=0.35, edgecolors="none", label="_all")

    for md in modes:
        sel_m = (m==md)
        for gamma in gammas:
            sel = sel_m & (g==gamma)
            sel_front = np.zeros_like(sel, bool); sel_front[front_indices]=True
            sel &= sel_front
            if not sel.any(): continue
            plt.scatter(x[sel], z[sel],
                        s=60.0*(1.0+pM[sel]),
                        marker=marks[md], facecolors=color[gamma],
                        edgecolors="k", linewidths=0.7,
                        label=f"Pareto: {md}, γ={gamma}")
            for xi, yi, di in zip(x[sel], z[sel], d[sel]):
                plt.annotate(f"d={di:.2f}", (xi, yi), textcoords="offset points", xytext=(4,4), fontsize=8)
    # -------- Baseline 強調 --------
    baseline_mask = np.isclose(d, 0.33, atol=1e-6) & np.isclose(g, 1.0, atol=1e-6) & (m.astype(str) == "sppf")
    if baseline_mask.any():
        xi, yi = x[baseline_mask][0], z[baseline_mask][0]
        plt.scatter(xi, yi, s=100, c="red", marker="*", edgecolors="k", linewidths=1.2, label="Baseline")
        plt.annotate("Baseline (d=0.33, g=1.0, sppf)", (xi, yi),
                     xytext=(10,10), textcoords="offset points", fontsize=9, color="red")

    plt.xlabel(xlab); plt.ylabel("ZiCo (↑)")
    plt.title(f"Pareto Front — ZiCo vs {xlab} (color=gamma, marker=SPP mode, size=Params)")
    # dedupe legend
    h, lb = plt.gca().get_legend_handles_labels()
    uniq, seen = [], set()
    for hh, ll in zip(h, lb):
        if ll not in seen and not ll.startswith("_"):
            uniq.append((hh,ll)); seen.add(ll)
    if uniq: plt.legend([u[0] for u in uniq], [u[1] for u in uniq], fontsize=9, ncol=2, framealpha=0.9)
    plt.grid(True, alpha=0.3)
    out = os.path.join(PLOT_DIR, f"pareto_unified_{'flops' if x_metric=='flops' else 'params'}_{title_suffix}.png")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close(); print("saved:", out)

def plot_unified_with_pareto_all(recs, front_indices, x_metric="flops", title_suffix="all"):
    z = np.array([r[0] for r in recs], float)
    pM = np.array([r[1] for r in recs], float) / 1e6
    fG = np.array([(r[6] / 1e9) if r[6] is not None else np.nan for r in recs], float)
    d = np.array([r[3] for r in recs], float)
    g = np.array([r[4] for r in recs], float)
    m = np.array([r[5] for r in recs], object)

    if x_metric == "flops":
        x = fG; xlab = "FLOPs (G)"; base_mask = ~np.isnan(x)
    else:
        x = pM; xlab = "Params (M)"; base_mask = ~np.isnan(z)

    x_all, y_all = x[base_mask], z[base_mask]
    size_all = 20.0 * (1.0 + pM[base_mask])

    modes = ["none", "sppf", "spp"]
    marks = {"none": "o", "sppf": "^", "spp": "s"}
    gammas = sorted(set(g.tolist()))
    cmap = plt.get_cmap("tab10")
    color = {gammas[i]: cmap(i % 10) for i in range(len(gammas))}

    plt.figure(figsize=(8, 6))

    # ---- 全点を条件ごとに描画 ----
    for md in modes:
        sel_m = (m == md)
        for gamma in gammas:
            sel = sel_m & (g == gamma)
            if not sel.any():
                continue
            plt.scatter(x[sel], z[sel],
                        s=40.0 * (1.0 + pM[sel]),
                        marker=marks[md], facecolors=color[gamma],
                        edgecolors="none", alpha=0.6,
                        label=f"{md}, γ={gamma}")

            # 各点に depth 値を表示
            for xi, yi, di in zip(x[sel], z[sel], d[sel]):
                plt.annotate(f"d={di:.2f}", (xi, yi),
                             textcoords="offset points", xytext=(3, 3),
                             fontsize=7, alpha=0.8)

    # ---- Pareto front の点だけ黒枠を付けて強調 ----
    if len(front_indices) > 0:
        for idx in front_indices:
            plt.scatter(x[idx], z[idx],
                        s=60.0 * (1.0 + pM[idx]),
                        marker=marks[m[idx]], facecolors=color[g[idx]],
                        edgecolors="k", linewidths=1.0)

    # ---- Baseline 強調 ----
    baseline_mask = np.isclose(d, 0.33, atol=1e-6) & np.isclose(g, 1.0, atol=1e-6) & (m.astype(str) == "sppf")
    if baseline_mask.any():
        xi, yi = x[baseline_mask][0], z[baseline_mask][0]
        plt.scatter(xi, yi, s=300, c="red", marker="*", edgecolors="k", linewidths=1.2, label="Baseline")
        plt.annotate("Baseline", (xi, yi),
                     xytext=(10, 10), textcoords="offset points", fontsize=9, color="red")

    plt.xlabel(xlab); plt.ylabel("ZiCo (↑)")
    plt.title(f"ZiCo vs {xlab} (all points, color=gamma, marker=SPP mode, size=Params)")

    # ---- legend dedupe ----
    h, lb = plt.gca().get_legend_handles_labels()
    uniq, seen = [], set()
    for hh, ll in zip(h, lb):
        if ll not in seen and not ll.startswith("_"):
            uniq.append((hh, ll)); seen.add(ll)
    if uniq:
        plt.legend([u[0] for u in uniq], [u[1] for u in uniq],
                   fontsize=9, ncol=2, framealpha=0.9)

    plt.grid(True, alpha=0.3)
    out = os.path.join(PLOT_DIR, f"all_with_d_{'flops' if x_metric=='flops' else 'params'}_{title_suffix}.png")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print("saved:", out)


def save_pareto_csv(recs, front_indices, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ZiCo","Params","Latency_us","Depth","Gamma","SPPmode","FLOPs"])
        for i in front_indices:
            z,p,lat,d,g,md,fl = recs[i]
            w.writerow([z,p,lat,d,g,md,fl])
    print("saved:", path)



def objective_lat(trial):
    d     = trial.suggest_categorical("depth", DEPTH_CAND)
    gamma = trial.suggest_categorical("gamma", GAMMA_CAND)
    mode  = trial.suggest_categorical("spp_mode", SPP_MODE_CAND)
    w     = DEPTH_WIDTH_MAP[d]

    with open(YAML_PATH) as f:
        cfg = yaml.safe_load(f)
    cfg = apply_depth_width(cfg, d, w)
    cfg = set_spp_mode_in_cfg(cfg, mode=mode)
    cfg = set_expansion(cfg, E_FIXED)
    cfg = set_activation(cfg, ACT_FIXED)

    model = Model(cfg).to(device)

    # gamma スケール
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) and m.weight is not None:
                m.weight.mul_(float(gamma))

    dummy = torch.randn(1, 3, 640, 640, device=device)
    zico, params, lat_us, flops = compute_model_metrics(model, dummy)

    results.append((zico, params, lat_us, d, gamma, mode, flops))

    print(f"[Trial] d={d:.2f}, w={w:.2f}, mode={mode}, γ={gamma} "
          f"=> ZiCo={zico:.2f}, Params={params/1e6:.2f}M, Lat={lat_us/1e3:.2f}ms, FLOPs(G)={(flops/1e9) if flops else None}")

    # Optuna には ZiCo を maximize, Latency を minimize の2目的で返す
    return zico, lat_us

import numpy as np
import matplotlib.pyplot as plt
import os

def plot_zico_latency_unified(recs, front_indices, title_suffix="zico_latency"):
    """
    recs: (zico, params, lat_us, depth, gamma, mode, flops)
    front_indices: list/array of indices on Pareto front
    """

    z = np.array([r[0] for r in recs], float)
    pM = np.array([r[1] for r in recs], float) / 1e6
    lat_ms = np.array([r[2] for r in recs], float) / 1e3  # us→ms
    d  = np.array([r[3] for r in recs], float)
    g  = np.array([r[4] for r in recs], float)
    m  = np.array([r[5] for r in recs], object)

    # 全点
    plt.figure(figsize=(8,6))
    plt.scatter(lat_ms, z, s=20*(1.0+pM), c="#bbbbbb", alpha=0.35,
                edgecolors="none", label="_all")

    # マーカーと色
    modes = ["none","sppf","spp"]
    marks = {"none":"o","sppf":"^","spp":"s"}
    gammas = sorted(set(g.tolist()))
    cmap = plt.get_cmap("tab10")
    color = {gammas[i]: cmap(i%10) for i in range(len(gammas))}

    # パレート点だけ強調
    for md in modes:
        sel_m = (m==md)
        for gamma in gammas:
            sel = sel_m & (g==gamma)
            sel_front = np.zeros_like(sel, bool); sel_front[front_indices]=True
            sel &= sel_front
            if not sel.any(): continue
            plt.scatter(lat_ms[sel], z[sel],
                        s=60*(1.0+pM[sel]),
                        marker=marks[md], facecolors=color[gamma],
                        edgecolors="k", linewidths=0.7,
                        label=f"Pareto: {md}, γ={gamma}")
            for xi, yi, di in zip(lat_ms[sel], z[sel], d[sel]):
                plt.annotate(f"d={di:.2f}", (xi, yi),
                             textcoords="offset points", xytext=(4,4), fontsize=8)

    # Baseline
    baseline_mask = (d==0.33) & (g==1.0) & (m=="sppf")
    if baseline_mask.any():
        xb, yb = lat_ms[baseline_mask][0], z[baseline_mask][0]
        plt.scatter(xb, yb, s=100, c="red", marker="*", edgecolors="k", linewidths=1.2, label="Baseline")
        plt.annotate("Baseline (d=0.33, g=1.0, sppf)", (xb, yb),
                     xytext=(10,10), textcoords="offset points", fontsize=9, color="red")

    plt.xlabel("Latency (ms) ↓ smaller is better")
    plt.ylabel("ZiCo (↑)")
    plt.title("Pareto Front — ZiCo vs Latency (color=gamma, marker=SPP mode, size=Params)")

    # legend dedupe
    h, lb = plt.gca().get_legend_handles_labels()
    uniq, seen = [], set()
    for hh, ll in zip(h, lb):
        if ll not in seen and not ll.startswith("_"):
            uniq.append((hh,ll)); seen.add(ll)
    if uniq: plt.legend([u[0] for u in uniq], [u[1] for u in uniq],
                        fontsize=9, ncol=2, framealpha=0.9)

    plt.grid(True, alpha=0.3)
    out = os.path.join(PLOT_DIR, f"pareto_unified_latency_{title_suffix}.png")
    plt.tight_layout(); plt.savefig(out, dpi=300); plt.close()
    print("saved:", out)




if __name__ == "__main__":
    repeats = 2  # サンプル数を増やしたいなら上げる
    n_trials = len(DEPTH_CAND)*len(GAMMA_CAND)*len(SPP_MODE_CAND)*repeats

    # """flops/params vs ZiCo"""
    study = optuna.create_study(directions=["maximize","minimize","minimize"],
                                sampler=NSGAIISampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # パレート抽出・可視化
    front_idx = compute_pareto_front(results)
    plot_unified_with_pareto(results, front_indices=front_idx, x_metric="flops",  title_suffix="spp_gamma_depth")
    plot_unified_with_pareto(results, front_indices=front_idx, x_metric="params", title_suffix="spp_gamma_depth")
    plot_unified_with_pareto_all(results, front_indices=front_idx, x_metric="flops",  title_suffix="spp_gamma_depth")
    plot_unified_with_pareto_all(results, front_indices=front_idx, x_metric="params", title_suffix="spp_gamma_depth")
    #save_pareto_csv(results, front_idx, os.path.join(PLOT_DIR, "pareto_spp_gamma_depth.csv"))
    study = optuna.create_study(
    directions=["maximize", "minimize", "minimize"],
    sampler=NSGAIISampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # study = optuna.create_study(
    # directions=["maximize", "minimize"],  # ZiCo ↑, Latency ↓
    # sampler=NSGAIISampler(seed=42),
    # )
    # study.optimize(objective_lat, n_trials=n_trials, show_progress_bar=True)
    # front_idx = compute_pareto_front(results)
    # plot_zico_latency_unified(results, front_indices=front_idx, title_suffix="spp_gamma_depth")
