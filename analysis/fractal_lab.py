#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分形方法对比实验（因子建模师 × 小波顾问）

模板：下跌 → N 字上涨 → N 字回落不创新低 → 突破（120 根 60 分钟）
方法：M1 Higuchi FD / M2 MF-DFA 谱宽 / M3 R/S Hurst / M4 小波跨尺度相似 /
      M5 DTW 距离 / M6 形态符号编码 / M0 已实现波动（对照）

用法：
  python fractal_lab.py --selftest
  python fractal_lab.py --run
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis')
from pattern_similarity import read60, wave_features, wave_similarity

SRC = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/tmp60/60分钟'
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
W = 120          # 模板窗口（30 个交易日）
HOLD = 80        # 20 个交易日 = 80 根
LABEL_TH = 0.30


def build_template():
    seg1 = np.linspace(1.00, 0.88, 40)   # 下跌
    seg2 = np.linspace(0.88, 1.10, 40)   # N 字上涨
    seg3 = np.linspace(1.10, 1.02, 30)   # N 字回落，不创新低
    seg4 = np.linspace(1.02, 1.08, 10)   # 突破
    return np.concatenate([seg1, seg2, seg3, seg4])


TPL = build_template()


def _norm(x):
    x = np.asarray(x, float)
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-12 else 1.0)


# ---------------------------------------------------------------- 方法

def path_roughness(x):
    """路径长度 / 首尾距离。直线=1，越弯曲越大。

    说明：严格的分形维数（Higuchi / box-counting）在 120 点窗口下估计不稳定
    （点数少于 ε² 会饱和），这里用无歧义的粗糙度代理，并在报告中注明口径。
    """
    x = _norm(x)
    return float(np.sum(np.abs(np.diff(x))) / (abs(x[-1] - x[0]) + 1e-9))


def mfdfa_width(x, qs=(-3, -1, 1, 3), scales=(8, 16, 32, 64)):
    x = np.asarray(x, float)
    y = np.cumsum(x - x.mean())
    Fs, used = [], []
    for s in scales:
        ns = len(y) // s
        if ns < 2:
            continue
        segs = y[:ns * s].reshape(ns, s)
        t = np.arange(s)
        fq = []
        for q in qs:
            f = []
            for seg in segs:
                coef = np.polyfit(t, seg, 1)
                res = seg - np.polyval(coef, t)
                f.append(np.mean(res ** 2))
            f = np.maximum(np.array(f), 1e-12)
            fq.append(np.exp(0.5 * np.mean(np.log(f))) if abs(q) < 1e-9
                      else np.mean(f ** (q / 2)) ** (1 / q))
        Fs.append(fq)
        used.append(s)
    if len(used) < 2:
        return np.nan
    h = []
    for j in range(len(qs)):
        col = np.array([Fs[i][j] for i in range(len(used))])
        h.append(np.polyfit(np.log(used), np.log(col), 1)[0])
    return float(max(h) - min(h))


def hurst_rs(x, min_n=8):
    x = np.asarray(x, float)
    N = len(x)
    ns, rs = [], []
    n = min_n
    while n <= N // 2:
        k = N // n
        vals = []
        for i in range(k):
            seg = x[i * n:(i + 1) * n]
            dev = np.cumsum(seg - seg.mean())
            R = dev.max() - dev.min()
            S = seg.std(ddof=0)
            if S > 1e-12:
                vals.append(R / S)
        if vals:
            ns.append(n)
            rs.append(np.mean(vals))
        n *= 2
    if len(ns) < 2:
        return np.nan
    return float(np.polyfit(np.log(ns), np.log(rs), 1)[0])


def dtw_dist(a, b):
    a, b = _norm(a), _norm(b)
    n, m = len(a), len(b)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = abs(a[i - 1] - b[j - 1]) + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return float(D[n, m])


def shape_code(x, k=8):
    seg = np.array_split(np.asarray(x, float), k)
    means = np.array([s.mean() for s in seg])
    return np.sign(np.diff(means))


def code_match(a, b):
    return float(np.mean(a == b))


def features(seg):
    seg = np.asarray(seg, float)
    rets = np.diff(np.log(np.maximum(seg, 1e-12)))
    return {
        "M1_rough": path_roughness(seg),
        "M2_mfdfa": mfdfa_width(rets),
        "M3_hurst": hurst_rs(rets),
        "M4_wave": wave_similarity(wave_features(seg), wave_features(TPL)),
        "M5_dtw": -dtw_dist(seg, TPL),
        "M6_code": code_match(shape_code(seg), shape_code(TPL)),
        "M0_vol": float(np.std(rets)),
    }


# ---------------------------------------------------------------- 评估

def auc_score(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def run():
    ev = json.load(open(os.path.join(A, "st_events.json"), encoding="utf-8"))
    new_st = {k: v for k, v in ev.items() if v["st_date"]}
    rows = []
    for code in new_st:
        p = os.path.join(SRC, code + ".csv")
        if not os.path.exists(p):
            continue
        bars, closes = read60(p)
        if len(bars) < W + HOLD + 5:
            continue
        idx = next((i for i, b in enumerate(bars) if "2025-10-15" <= b[0] <= "2025-11-07"), None)
        if idx is None or idx < W - 1 or idx + HOLD >= len(closes):
            continue
        seg = closes[idx - W + 1: idx + 1]
        f = features(seg)
        ret = closes[idx + HOLD] / closes[idx] - 1
        rows.append({"code": code, "ret": ret, "label": int(ret >= LABEL_TH), **f})

    n_pos = sum(r["label"] for r in rows)
    print(f"样本 {len(rows)} 只 | 正样本(≥{LABEL_TH:.0%}) {n_pos} 只 | 负样本 {len(rows)-n_pos} 只\n")

    header = f"{'方法':<12}{'Spearman':>10}{'p值':>8}{'AUC':>8}{'Top10命中':>10}{'结论':>12}"
    print(header)
    print("-" * len(header))
    result = []
    for m in ["M1_rough", "M2_mfdfa", "M3_hurst", "M4_wave", "M5_dtw", "M6_code", "M0_vol"]:
        xs = [r[m] for r in rows]
        ys = [r["ret"] for r in rows]
        if any(np.isnan(xs)):
            print(f"{m:<12} 含 NaN，跳过")
            continue
        rho, p = spearmanr(xs, ys)
        auc = auc_score(xs, [r["label"] for r in rows])
        top = sorted(rows, key=lambda r: -r[m])[:10]
        hit = sum(r["label"] for r in top) / 10
        verdict = "有效" if (abs(rho) > 0.25 and p < 0.05) else ("弱" if abs(rho) > 0.15 else "无效")
        print(f"{m:<12}{rho:>+10.3f}{p:>8.3f}{auc:>8.3f}{hit:>10.0%}{verdict:>12}")
        result.append((m, rho, p, auc, hit, verdict))
    se = np.sqrt(0.25 / max(n_pos, 1) + 0.25 / max(len(rows) - n_pos, 1))
    print(f"\nAUC 的 95% 区间约 ±{1.96*se:.2f}（Hanley-McNeil 近似；正样本仅 {n_pos} 只，区间很宽）")
    print("判读：区间跨过 0.5 的，AUC 不能视为有效。")
    return rows, result


# ---------------------------------------------------------------- 自检

def selftest():
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'OK ' if cond else 'FAIL'}] {name} {info}")

    # 1) 模板自匹配
    chk("DTW 自距离=0", abs(dtw_dist(TPL, TPL)) < 1e-9)
    chk("小波自相似≈1", abs(wave_similarity(wave_features(TPL), wave_features(TPL)) - 1) < 1e-9)
    chk("形态编码自匹配=1", code_match(shape_code(TPL), shape_code(TPL)) == 1.0)

    # 2) 路径粗糙度：直线=1，白噪声远大于 1
    line = np.linspace(0, 1, 200)
    chk("粗糙度 直线=1", abs(path_roughness(line) - 1.0) < 1e-6, f"{path_roughness(line):.3f}")
    rng0 = np.random.default_rng(3)
    white = rng0.normal(size=200)
    chk("粗糙度 白噪声>5", path_roughness(white) > 5, f"{path_roughness(white):.1f}")

    # 3) Hurst：白噪声 ≈0.5（对收益率算，不是对价格）
    rng = np.random.default_rng(7)
    h = hurst_rs(rng.normal(size=500))
    chk("Hurst 白噪声≈0.5", 0.3 < h < 0.75, f"{h:.3f}")

    # 4) MF-DFA：收益率序列谱宽有限
    w = mfdfa_width(rng.normal(size=300))
    chk("MF-DFA 谱宽有限", np.isfinite(w) and w >= 0, f"{w:.3f}")

    # 5) 反向模板的 DTW 距离应大于自距离
    chk("反向模板距离更大", dtw_dist(TPL[::-1], TPL) > dtw_dist(TPL, TPL) + 1e-6)

    print("\n自检结果:", "全部通过 ✅" if ok else "存在失败 ❌")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    if args.run:
        run()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
