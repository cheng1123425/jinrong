# -*- coding: utf-8 -*-
"""分形方法 · 日线版样本外检验（2024 为样本外，2025 为样本内）"""
import sys, pickle
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis')
from fractal_lab import (path_roughness, mfdfa_width, hurst_rs, dtw_dist,
                         shape_code, code_match, TPL, auc_score)
from pattern_similarity import wave_features, wave_similarity

A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = pickle.load(open(A + '/panel2425.pkl', 'rb'))
W, HOLD, TH = 120, 20, 0.30

def st_events(panel, year):
    res = {}
    for code, d in panel.items():
        idx = [i for i, dt in enumerate(d['dates']) if dt[:4] == str(year)]
        if not idx or 'ST' in d['name'][idx[0]]:
            continue
        prev = False
        for i in idx:
            cur = 'ST' in d['name'][i]
            if cur and not prev:
                res[code] = i
                break
            prev = cur
    return res

def feats(seg):
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

def collect(year):
    ev = st_events(panel, year)
    rows = []
    for code in ev:
        d = panel[code]
        idx = next((i for i, dt in enumerate(d['dates'])
                    if f'{year}-10-15' <= dt <= f'{year}-11-07'), None)
        if idx is None or idx < W - 1 or idx + HOLD >= len(d['c']):
            continue
        seg = d['c'][idx - W + 1: idx + 1]
        if min(seg) <= 0:
            continue
        f = feats(seg)
        ret = d['c'][idx + HOLD] / d['c'][idx] - 1
        rows.append({"code": code, "ret": ret, "label": int(ret >= TH), **f})
    return rows

def evaluate(rows, title):
    n_pos = sum(r['label'] for r in rows)
    print(f"\n===== {title} | 样本 {len(rows)} | 正样本 {n_pos} =====")
    print(f"{'方法':<11}{'Spearman':>10}{'p':>8}{'AUC':>8}{'Top10':>8}{'判定':>8}")
    out = []
    for m in ["M1_rough","M2_mfdfa","M3_hurst","M4_wave","M5_dtw","M6_code","M0_vol"]:
        xs = [r[m] for r in rows]; ys = [r['ret'] for r in rows]
        if any(np.isnan(xs)) or len(set(xs)) < 3:
            continue
        rho, p = spearmanr(xs, ys)
        auc = auc_score(xs, [r['label'] for r in rows])
        top = sorted(rows, key=lambda r: -r[m])[:10]
        hit = sum(r['label'] for r in top) / 10
        verdict = "有效" if (abs(rho) > 0.25 and p < 0.05) else ("弱" if p < 0.05 else "无效")
        print(f"{m:<11}{rho:>+10.3f}{p:>8.3f}{auc:>8.3f}{hit:>8.0%}{verdict:>8}")
        out.append((m, rho, p, auc))
    return out

r24 = collect(2024); r25 = collect(2025)
evaluate(r24, "2024（样本外）")
evaluate(r25, "2025（样本内）")
evaluate(r24 + r25, "2024+2025 合并")
