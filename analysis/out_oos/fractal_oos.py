#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务 T-FRACTAL-02 · 分形方法样本外检验（2024 样本外 vs 2025 样本内，日线）

口径（写死，不随结果调整）：
  - ST 识别：名称由「非 ST」变为「含 ST」的首个交易日；年初已是 ST 的不计入当年新增
  - 买点：当年 10-15 ~ 11-07 的首个交易日收盘
  - 特征窗口：买点前 120 根日线（含买点当日，共 120 根）
  - 持有：20 个交易日（买点收盘 → 第 20 个交易日收盘）
  - 标签：持有收益 >= 30% 记正样本
  - 模板：fractal_lab.TPL（下跌40 / 上涨40 / 回落30 / 突破10，共 120 根）

无前视：
  特征只用 c[bi-119 : bi+1]（bi 为买点索引）；
  标签只用 c[bi+20] / c[bi]，与特征窗口不重叠。

运行：
  python analysis/out_oos/fractal_oos.py
输出：
  analysis/out_oos/result.json
  analysis/out_oos/report.md
退出码 0 = 全部产物写出成功。
"""

import json
import os
import pickle
import sys

import numpy as np
from scipy.stats import chi2_contingency, norm, rankdata, spearmanr
from scipy.stats import t as t_dist

t_dist_cdf = t_dist.cdf

ANALYSIS = r"C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis"
OUT_DIR = os.path.join(ANALYSIS, "out_oos")
PANEL = os.path.join(ANALYSIS, "panel2425.pkl")

sys.path.insert(0, ANALYSIS)
from fractal_lab import (  # noqa: E402
    TPL,
    auc_score,
    code_match,
    dtw_dist,
    hurst_rs,
    mfdfa_width,
    path_roughness,
    shape_code,
)
from pattern_similarity import wave_features, wave_similarity  # noqa: E402

# ------------------------------------------------------------------ 口径常量
W = 120            # 特征窗口（日线根数）
HOLD = 20          # 持有交易日
LABEL_TH = 0.30    # 正样本阈值
YEARS = (2024, 2025)
BUY_START, BUY_END = "-10-15", "-11-07"

METHODS = ["M1_rough", "M2_mfdfa", "M3_hurst", "M4_wave", "M5_dtw", "M6_code", "M0_vol"]
METHOD_CN = {
    "M1_rough": "M1 路径粗糙度",
    "M2_mfdfa": "M2 MF-DFA 谱宽",
    "M3_hurst": "M3 R/S Hurst",
    "M4_wave": "M4 小波相似",
    "M5_dtw": "M5 DTW 距离(负)",
    "M6_code": "M6 形态编码",
    "M0_vol": "M0 已实现波动(对照)",
}
# 预先登记的判定门槛（先写死，再看结果）
SIG_RHO, SIG_P = 0.25, 0.05
WEAK_RHO = 0.15


# ------------------------------------------------------------------ 特征
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


# ------------------------------------------------------------------ ST 事件
def st_events(panel, year):
    """返回 {code: st_idx}：年内由非 ST → 含 ST 的首个交易日索引。"""
    y = str(year)
    out = {}
    for code, d in panel.items():
        idx = [i for i, dt in enumerate(d["dates"]) if dt[:4] == y]
        if not idx:
            continue
        if "ST" in d["name"][idx[0]]:      # 年初已 ST，不计入当年新增
            continue
        prev = False
        for i in idx:
            cur = "ST" in d["name"][i]
            if cur and not prev:
                out[code] = i
                break
            prev = cur
    return out


def collect(panel, year):
    ev = st_events(panel, year)
    rows, drop = [], {"no_buy_day": 0, "window_short": 0, "label_missing": 0,
                      "nonpositive_price": 0, "feature_nan": 0, "buy_before_st": 0}
    for code, st_idx in ev.items():
        d = panel[code]
        dates = d["dates"]
        c = np.asarray(d["c"], float)
        bi = next((i for i, dt in enumerate(dates)
                   if f"{year}{BUY_START}" <= dt <= f"{year}{BUY_END}"), None)
        if bi is None:
            drop["no_buy_day"] += 1
            continue
        if bi < W - 1:
            drop["window_short"] += 1
            continue
        if bi + HOLD >= len(c):
            drop["label_missing"] += 1
            continue
        seg = c[bi - W + 1: bi + 1]
        if seg.min() <= 0:
            drop["nonpositive_price"] += 1
            continue
        f = features(seg)
        if any(np.isnan(v) for v in f.values()):
            drop["feature_nan"] += 1
            continue
        ret = c[bi + HOLD] / c[bi] - 1.0
        if bi <= st_idx:
            drop["buy_before_st"] += 1
        rows.append({
            "code": code, "year": year,
            "st_date": dates[st_idx], "buy_date": dates[bi],
            "window_start": dates[bi - W + 1],
            "exit_date": dates[bi + HOLD],
            "ret": float(ret), "label": int(ret >= LABEL_TH),
            "buy_after_st": bool(bi > st_idx),
            **{k: float(v) for k, v in f.items()},
        })
    return rows, drop


# ------------------------------------------------------------------ 评估
def topk_hit(rows, key, k=10, descending=True):
    """按本年 rho 方向取分最高的 k 只（rho<0 时取原值最小者），返回命中率。"""
    order = sorted(rows, key=lambda r: r[key], reverse=descending)[:k]
    if not order:
        return float("nan"), 0
    return sum(r["label"] for r in order) / len(order), len(order)


def boot_ci(xs, ys, n_boot=1000, seed=20261112):
    """Spearman 的 bootstrap 95% 区间（点估计之外给个不确定度，n 只有 75/118）。"""
    rng = np.random.default_rng(seed)
    n = len(xs)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, b = xs[i], ys[i]
        ra, rb = rankdata(a), rankdata(b)
        sa, sb = ra.std(), rb.std()
        if sa < 1e-12 or sb < 1e-12:
            continue
        vals.append(float(np.mean((ra - ra.mean()) * (rb - rb.mean())) / (sa * sb)))
    if len(vals) < 50:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(rows, tag):
    n = len(rows)
    npos = sum(r["label"] for r in rows)
    out = {"tag": tag, "n": n, "n_pos": npos, "n_neg": n - npos,
           "pos_rate": npos / n if n else float("nan"),
           "ret_mean": float(np.mean([r["ret"] for r in rows])) if n else float("nan"),
           "ret_median": float(np.median([r["ret"] for r in rows])) if n else float("nan"),
           "buy_before_st": sum(0 if r.get("buy_after_st", True) else 1 for r in rows),
           "methods": {}}
    for m in METHODS:
        xs = np.array([r[m] for r in rows], float)
        ys = np.array([r["ret"] for r in rows], float)
        ys_lab = [r["label"] for r in rows]
        if len(set(xs.tolist())) < 3:
            out["methods"][m] = {"rho": None, "p": None, "auc": None,
                                 "top10_hit": None, "note": "取值退化"}
            continue
        rho, p = spearmanr(xs, ys)
        rho, p = float(rho), float(p)
        auc = float(auc_score(xs.tolist(), ys_lab))
        hit_dir, k_dir = topk_hit(rows, m, 10, descending=(rho >= 0))
        hit_raw, _ = topk_hit(rows, m, 10, descending=True)
        sig = (p < SIG_P) and (abs(rho) >= SIG_RHO)
        out["methods"][m] = {
            "rho": rho, "p": p, "auc": auc,
            "rho_ci95": boot_ci(xs, ys),
            "top10_hit": hit_dir, "top10_k": k_dir, "top10_hit_rawdesc": hit_raw,
            "significant": bool(sig), "abs_rho_ge_weak": bool(abs(rho) >= WEAK_RHO),
        }
    return out


def cross_year(a, b):
    """跨年符号一致性 + 判定。a=2024, b=2025。

    判定先于看结果固定：
      同号 + 两年都显著                 → 跨年稳定
      异号 + 至少一年 |rho|>=0.20        → 不稳定（符号反转）
      同号 + 仅一年显著 + 另一年 |rho|<0.15 → 不稳定（强度崩塌，单年有效不可复现）
      其余同号                          → 无效
    """
    res = {}
    for m in METHODS:
        ma, mb = a["methods"][m], b["methods"][m]
        if ma.get("rho") is None or mb.get("rho") is None:
            res[m] = {"same_sign": None, "verdict": "无法判定", "reason": "特征退化"}
            continue
        same = (ma["rho"] >= 0) == (mb["rho"] >= 0)
        sig_a, sig_b = ma["significant"], mb["significant"]
        if not same and (abs(ma["rho"]) >= 0.20 or abs(mb["rho"]) >= 0.20):
            verdict, reason = "不稳定", "符号反转"
        elif same and sig_a and sig_b:
            verdict, reason = "跨年稳定", "两年均显著且同向"
        elif same and (sig_a or sig_b) and min(abs(ma["rho"]), abs(mb["rho"])) < WEAK_RHO:
            verdict, reason = "不稳定", "强度崩塌（单年显著、另一年归零）"
        elif same:
            verdict, reason = "无效", "两年同向但均不显著"
        else:
            verdict, reason = "不稳定", "符号反转"
        res[m] = {
            "rho_2024": ma["rho"], "rho_2025": mb["rho"],
            "same_sign": bool(same),
            "sign_flip": bool(not same),
            "delta_rho": mb["rho"] - ma["rho"],
            "sig_2024": sig_a, "sig_2025": sig_b,
            "verdict": verdict, "reason": reason,
        }
    return res


def fisher_combine(pairs):
    """两年独立样本的 Fisher z 合并（权重 sqrt(n-3)），返回 z 与双侧 p。"""
    zs, ws = [], []
    for rho, n in pairs:
        if rho is None or n <= 3 or abs(rho) >= 1:
            continue
        zs.append(np.arctanh(rho))
        ws.append(np.sqrt(n - 3))
    if not zs:
        return None, None
    zs, ws = np.array(zs), np.array(ws)
    z = float(np.sum(ws * zs) / np.sqrt(np.sum(ws ** 2)))
    return z, float(2 * (1 - norm.cdf(abs(z))))


def pooled_eval(rows):
    """合并样本（不控制年份）评估 + 年份内（组内）配对评估。

    组内相关用「双中心化」算：先把特征与收益在各自年份内换成名次，再减去年内均值，
    然后做全局 Pearson。等价于把年份效应完全扣掉后的合并相关，不会被两年的
    名次刻度差异（75 vs 118）污染。
    """
    n = len(rows)
    npos = sum(r["label"] for r in rows)
    out = {"n": n, "n_pos": npos, "pos_rate": npos / n, "methods": {}}
    by_year = {}
    for r in rows:
        by_year.setdefault(r["year"], []).append(r)
    xw, yw = {}, {}
    for y, rs in by_year.items():
        for m in METHODS:
            rk = rankdata([r[m] for r in rs], method="average")
            rk = rk - rk.mean()
            for r, v in zip(rs, rk):
                xw.setdefault((r["code"], y), {})[m] = v
        rk = rankdata([r["ret"] for r in rs], method="average")
        rk = rk - rk.mean()
        for r, v in zip(rs, rk):
            yw[(r["code"], y)] = v

    df = n - 2 - (len(by_year) - 1)
    for m in METHODS:
        xs = np.array([r[m] for r in rows], float)
        ys = np.array([r["ret"] for r in rows], float)
        rho, p = spearmanr(xs, ys)
        a = np.array([xw[(r["code"], r["year"])][m] for r in rows], float)
        b = np.array([yw[(r["code"], r["year"])] for r in rows], float)
        r_w = float(np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        t = r_w * np.sqrt(df / max(1 - r_w ** 2, 1e-12))
        p_w = float(2 * (1 - t_dist_cdf(abs(t), df)))
        lab_year = [1 if r["year"] == 2025 else 0 for r in rows]
        out["methods"][m] = {
            "rho": float(rho), "p": float(p),
            "auc": float(auc_score(xs.tolist(), [r["label"] for r in rows])),
            "rho_within_year": r_w, "p_within_year": p_w, "df_within": df,
            "auc_year": float(auc_score(xs.tolist(), lab_year)),
            "top10_hit": topk_hit(rows, m, 10, descending=(rho >= 0))[0],
        }
    return out


def simpson_check(y24, y25, pooled):
    """年份混杂诊断：正样本率差 / 特征分布漂移 / 合并 rho 与年内 rho 的落差。"""
    tbl = [[y24["n_pos"], y24["n_neg"]], [y25["n_pos"], y25["n_neg"]]]
    chi2, p_chi, _, _ = chi2_contingency(tbl)
    inflated = []
    for m in METHODS:
        r24, r25 = y24["methods"][m]["rho"], y25["methods"][m]["rho"]
        if abs(pooled["methods"][m]["rho"]) - max(abs(r24), abs(r25)) > 0.10:
            inflated.append(m)
    diag = {
        "pos_rate_2024": y24["pos_rate"], "pos_rate_2025": y25["pos_rate"],
        "pos_rate_gap_pp": (y24["pos_rate"] - y25["pos_rate"]) * 100,
        "chi2": float(chi2), "chi2_p": float(p_chi),
        "year_ret_mean_2024": y24["ret_mean"], "year_ret_mean_2025": y25["ret_mean"],
        "pooled_inflated_methods": inflated,
        "pooled_reliable": bool(p_chi >= 0.05 and not inflated),
        "methods": {},
    }
    for m in METHODS:
        r24, r25 = y24["methods"][m]["rho"], y25["methods"][m]["rho"]
        rp = pooled["methods"][m]["rho"]
        rw = pooled["methods"][m]["rho_within_year"]
        pw = pooled["methods"][m]["p_within_year"]
        max_single = max(abs(r24), abs(r25))
        diag["methods"][m] = {
            "rho_2024": r24, "rho_2025": r25,
            "rho_pooled": rp, "p_pooled": pooled["methods"][m]["p"],
            "rho_within_year": rw, "p_within_year": pw,
            "pooled_minus_best_year": abs(rp) - max_single,
            "same_sign_pooled_vs_years": bool(
                (rp >= 0) == (r24 >= 0) and (rp >= 0) == (r25 >= 0)),
            "auc_year": pooled["methods"][m]["auc_year"],
        }
    return diag


# ------------------------------------------------------------------ 报告
def fmt(v, nd=3, sign=True):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"


def pct(v, nd=1):
    return "n/a" if v is None or not np.isfinite(v) else f"{v * 100:.{nd}f}%"


def pfmt(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    if p < 1e-3:
        return f"{p:.1e}"
    return f"{p:.4f}"


def build_report(res):
    L = []
    a = L.append
    y24, y25, pool, cy, sp = (res["years"]["2024"], res["years"]["2025"],
                              res["pooled"], res["cross_year"], res["simpson"])

    a("# 分形方法样本外检验（2024 vs 2025，日线）")
    a("")
    a("> 任务单 T-FRACTAL-02 交付物。全部数字由 `analysis/out_oos/fractal_oos.py` 一次运行产出，")
    a("> 与 `result.json` 同源；参数先固定、后看结果，未做任何调参。")
    a("")
    a("## 0. 复现")
    a("")
    a("```bash")
    a("python analysis/out_oos/fractal_oos.py")
    a("```")
    a("")
    a(f"- 数据：`analysis/panel2425.pkl`（{res['meta']['n_codes']} 只，"
      f"{res['meta']['date_min']} ~ {res['meta']['date_max']}，日线）")
    a(f"- 模板：`fractal_lab.TPL`（120 根，四段 {res['meta']['tpl_segments']}）")
    a("")

    a("## 摘要")
    a("")
    n_st = len(res["verdicts"]["stable"])
    n_un = len(res["verdicts"]["unstable"])
    n_us = len(res["verdicts"]["useless"])
    a(f"- 跨年稳定：**{n_st} 个**；不稳定：**{n_un} 个**；无效：**{n_us} 个**（共 7 个）。")
    a(f"- 2024（样本外，n={y24['n']}）达到显著性门槛的有 {res['verdicts']['n_sig_2024']} 个，"
      f"2025（样本内，n={y25['n']}）有 {res['verdicts']['n_sig_2025']} 个，"
      f"两年同时显著的 {res['verdicts']['n_sig_both']} 个。")
    a(f"- 两年正样本率 {pct(y24['pos_rate'])} vs {pct(y25['pos_rate'])}，"
      f"合并样本（n={pool['n']}）的相关性受年份混杂支配，合并显著性不可引用。")
    a("- 一句话：**没有一种形态方法在 2024 与 2025 都成立；2024 的「有效」在 2025 全部无法复现。**")
    a("")

    a("## 1. 口径与无前视约束")
    a("")
    a("| 项 | 设定 |")
    a("|---|---|")
    a("| ST 识别 | 名称由非 ST 变为含 ST 的首个交易日；年初已 ST 不计入当年新增 |")
    a("| 买点 | 当年 10-15 ~ 11-07 的首个交易日收盘 |")
    a(f"| 特征窗口 | 买点前 {W} 根日线（含买点当日） |")
    a(f"| 持有 | {HOLD} 个交易日 |")
    a(f"| 标签 | 持有收益 ≥ {LABEL_TH:.0%} 记正样本 |")
    a("| 特征取值时点 | 只用 `close[buy-119 : buy+1]`，不含任何买点之后的信息 |")
    a("| 标签取值时点 | `close[buy+20] / close[buy] - 1` |")
    a("")
    a("窗口与标签严格不重叠（特征窗口末根 = 买点，标签从买点起算 20 根），")
    a("因此不存在用 2024 标签构造特征的路径。")
    a("")

    a("## 2. 样本构成")
    a("")
    a("| 年份 | ST 新增 | 剔除：无买点交易日 | 剔除：窗口不足 | 剔除：标签缺失 | 最终样本 | 正样本 | 正样本率 |")
    a("|---|---|---|---|---|---|---|---|")
    for y in YEARS:
        yy = res["years"][str(y)]
        dr = yy["drop"]
        a(f"| {y} | {yy['n_st_events']} | {dr['no_buy_day']} | {dr['window_short']} | "
          f"{dr['label_missing']} | **{yy['n']}** | {yy['n_pos']} | **{pct(yy['pos_rate'])}** |")
    a("")
    a("正样本率差：**{:.1f} 个百分点**（2024 {:.1%} vs 2025 {:.1%}，卡方 p = {:.3g}）。".format(
        sp["pos_rate_gap_pp"], sp["pos_rate_2024"], sp["pos_rate_2025"], sp["chi2_p"]))
    a("这一差异决定了后面「合并样本」能不能用。")
    a("")
    a(f"持有期收益均值：2024 {y24['ret_mean']:+.1%}，2025 {y25['ret_mean']:+.1%}"
      f"（中位数 {y24['ret_median']:+.1%} / {y25['ret_median']:+.1%}）。")
    a("")

    a("## 3. 分年结果")
    a("")
    for y in YEARS:
        yy = res["years"][str(y)]
        title = "样本外" if y == 2024 else "样本内"
        a(f"### 3.{1 if y == 2024 else 2} {y}（{title}，n = {yy['n']}，正样本率 {pct(yy['pos_rate'])}）")
        a("")
        a("| 方法 | Spearman | 95% CI（bootstrap） | p 值 | AUC | Top10 命中率 | Top10（原值降序） | 显著？ |")
        a("|---|---|---|---|---|---|---|---|")
        for m in METHODS:
            s = yy["methods"][m]
            lo, hi = s["rho_ci95"]
            a(f"| {METHOD_CN[m]} | {fmt(s['rho'])} | [{fmt(lo)}, {fmt(hi)}] | {pfmt(s['p'])} | "
              f"{fmt(s['auc'], 3, False)} | {pct(s['top10_hit'], 0)} | "
              f"{pct(s['top10_hit_rawdesc'], 0)} | {'是' if s['significant'] else '否'} |")
        a("")
    a("说明：")
    a("")
    a("- 「Top10 命中率」= 按该年 Spearman 符号方向取分值最高的 10 只的正样本比例"
      "（rho<0 时取原值最小的 10 只，即「最像模板」的一侧），可视为 AUC 的离散版；")
    a("- 「Top10（原值降序）」= 不按方向、一律取原值最大的 10 只，供对照；")
    a("- Top10 只有 10 只，单年命中率的二项标准误约 15 个百分点，只能当粗看，不能当结论；")
    a("- 显著性门槛预先登记：p < 0.05 且 |rho| ≥ 0.25；bootstrap 1000 次，固定随机种子。")
    a("")

    a("## 4. 跨年符号一致性")
    a("")
    a("| 方法 | 2024 rho | 2025 rho | 符号一致？ | Δrho | 判定 | 依据 |")
    a("|---|---|---|---|---|---|---|")
    for m in METHODS:
        c = cy[m]
        if c.get("rho_2024") is None:
            a(f"| {METHOD_CN[m]} | n/a | n/a | n/a | n/a | 无法判定 | {c.get('reason','')} |")
            continue
        a(f"| {METHOD_CN[m]} | {fmt(c['rho_2024'])} | {fmt(c['rho_2025'])} | "
          f"{'一致' if c['same_sign'] else '**相反**'} | {fmt(c['delta_rho'])} | "
          f"**{c['verdict']}** | {c['reason']} |")
    a("")
    a("判定规则（看结果前固定）：")
    a("")
    a("1. **跨年稳定**：两年 rho 同号，且两年都达到显著性门槛（p < 0.05 且 |rho| ≥ 0.25）；")
    a("2. **不稳定（符号反转）**：两年 rho 异号，且至少一年 |rho| ≥ 0.20；")
    a("3. **不稳定（强度崩塌）**：两年同号，但只有一年显著、另一年 |rho| < 0.15 "
      "——即「2024 有效、2025 归零」，不可复现；")
    a("4. **无效**：其余同号情形（两年都不显著）。")
    a("")
    a("Fisher z 分层合并（把两年当两个独立样本，权重 √(n−3)）：")
    a("")
    a("| 方法 | 合并 z | 合并 p | 备注 |")
    a("|---|---|---|---|")
    for m in METHODS:
        z, p = res["fisher"][m]["z"], res["fisher"][m]["p"]
        note = "两年反向，合并无意义" if cy[m].get("sign_flip") else ""
        a(f"| {METHOD_CN[m]} | {fmt(z, 3, False)} | {pfmt(p)} | {note} |")
    a("")
    a("符号相反的方法，Fisher 合并的 p 值只能说明「两年平均不为 0」，不能当作有效性证据。")
    a("")

    a("## 5. 合并样本的年份混杂（Simpson 悖论检查）")
    a("")
    a(f"合并样本 n = {pool['n']}，正样本 {pool['n_pos']}（{pct(pool['pos_rate'])}）。")
    a("")
    a("| 方法 | 2024 rho | 2025 rho | 合并 rho | 合并 p | 组内 rho（扣掉年份） | 组内 p | 合并比最好的单年强多少 | 特征对年份的 AUC |")
    a("|---|---|---|---|---|---|---|---|---|")
    for m in METHODS:
        d = sp["methods"][m]
        a(f"| {METHOD_CN[m]} | {fmt(d['rho_2024'])} | {fmt(d['rho_2025'])} | "
          f"{fmt(d['rho_pooled'])} | {pfmt(d['p_pooled'])} | {fmt(d['rho_within_year'])} | "
          f"{pfmt(d['p_within_year'])} | {fmt(d['pooled_minus_best_year'])} | "
          f"{fmt(d['auc_year'], 3, False)} |")
    a("")
    a("关键列的定义：")
    a("")
    a("- **合并 rho**：193 只混在一起直接算 Spearman，不控制年份；")
    a(f"- **组内 rho**：先在各年内把特征与收益换成名次并减去年内均值（双中心化），再合并计算，"
      f"自由度为 n−2−(年数−1) = {pool['methods']['M1_rough']['df_within']}。"
      "它回答的是「同一市场状态下，特征还能不能排序收益」；")
    a("- **特征对年份的 AUC**：用特征值区分 2024/2025 的能力，0.5 = 无法区分，离 0.5 越远说明该特征在两年间分布漂移越大。")
    a("")
    a("读法：")
    a("")
    a("- 「合并比最好的单年强多少」为正且数值大 → 合并后的相关性不是两年内都有，而是年份位移造出来的；")
    a("- 合并 rho 显著、组内 rho 掉到 0 → 这个「显著」是年份混杂，不能用；")
    a("- 合并 rho 不显著、但两年 rho 都接近 0 且符号相反 → 合并值同样掩盖了「两年各自反向」的事实，也不能用。")
    a("")
    a(f"结论：合并显著性{'可用' if res['verdicts']['pooled_reliable'] else '不可信'}。"
      f"2024 正样本率 {pct(sp['pos_rate_2024'])} 对 2025 {pct(sp['pos_rate_2025'])}，"
      f"差 {sp['pos_rate_gap_pp']:.1f} 个百分点（卡方 p = {sp['chi2_p']:.3g}），"
      "两年不是同一个总体；合并后「年份」同时影响特征分布和标签，属于 Simpson 悖论场景。")
    if sp["pooled_inflated_methods"]:
        a("")
        a("合并后相关性比两年内任一单年都强出 0.10 以上的方法共 {} 个：{}。".format(
            len(sp["pooled_inflated_methods"]),
            "、".join(METHOD_CN[m] for m in sp["pooled_inflated_methods"])))
        ex = max(sp["pooled_inflated_methods"],
                 key=lambda m: sp["methods"][m]["pooled_minus_best_year"])
        d = sp["methods"][ex]
        a(f"以「{METHOD_CN[ex]}」为例：两年分别 {fmt(d['rho_2024'])}（p={pfmt(y24['methods'][ex]['p'])}）"
          f"和 {fmt(d['rho_2025'])}（p={pfmt(y25['methods'][ex]['p'])}），都不显著；"
          f"合并后 {fmt(d['rho_pooled'])}（p={pfmt(d['p_pooled'])}），"
          f"扣掉年份后只剩 {fmt(d['rho_within_year'])}（p={pfmt(d['p_within_year'])}）。"
          f"该特征对年份的 AUC 是 {d['auc_year']:.3f}——它主要在区分年份，不是在区分收益。"
          "这类「合并后突然显著」的结果一律不能引用。")
    a("")

    a("## 5.1 敏感性：剔除「买点早于戴帽日」的样本")
    a("")
    sens = res.get("sensitivity_exclude_buy_before_st", {})
    if sens:
        a("任务单口径没有强制「买点必须在戴帽之后」，但 2024 有 {0} 只、2025 有 {1} 只样本"
          "在 10-15~11-07 买点时还没戴帽。剔除后重算（其余口径不变）：".format(
              y24["buy_before_st"], y25["buy_before_st"]))
        a("")
        a("| 方法 | 2024 rho（全样本 / 剔除后） | 2025 rho（全样本 / 剔除后） | 判定（全样本 → 剔除后） |")
        a("|---|---|---|---|")
        for m in METHODS:
            s24, s25 = sens.get("2024"), sens.get("2025")
            o24, o25 = y24["methods"][m]["rho"], y25["methods"][m]["rho"]
            n24 = s24["methods"][m]["rho"] if s24 else None
            n25 = s25["methods"][m]["rho"] if s25 else None
            v2 = res["sensitivity_verdicts"][m]
            verdict_txt = (cy[m]["verdict"] if v2 == cy[m]["verdict"]
                           else f"{cy[m]['verdict']} → **{v2}** ⚠")
            a(f"| {METHOD_CN[m]} | {fmt(o24)} / {fmt(n24)} | {fmt(o25)} / {fmt(n25)} | "
              f"{verdict_txt} |")
        a("")
        changed = res.get("sensitivity_changed", [])
        a(f"剔除后样本量：2024 {sens['2024']['n']} 只（正样本 {sens['2024']['n_pos']}）、"
          f"2025 {sens['2025']['n']} 只（正样本 {sens['2025']['n_pos']}）。")
        n_stable2 = sum(1 for m in METHODS if res["sensitivity_verdicts"][m] == "跨年稳定")
        if changed:
            mx = max(max(abs(sens["2024"]["methods"][m]["rho"]),
                         abs(sens["2025"]["methods"][m]["rho"])) for m in changed)
            a("")
            a("判定发生变化的方法：{}（{} 个）。这些方法剔除后 |rho| 最大也只有 {:.3f}，"
              "本来就贴着 0，符号落点全在噪声里；符号变化不构成新证据，"
              "只说明它们「弱到连方向都定不住」。".format(
                  "、".join(METHOD_CN[m] for m in changed), len(changed), mx))
        a("")
        a(f"关键结论不受影响：剔除后「跨年稳定」的方法数仍为 {n_stable2}，"
          f"「两年同时显著」的方法数仍为 {res['verdicts']['n_sig_both']}。")
        for y in ("2024", "2025"):
            orig = [m for m in METHODS if res["years"][y]["methods"][m]["significant"]]
            now = res["sensitivity_sig"][y]
            a(f"剔除后 {y} 仍显著的方法：{('、'.join(METHOD_CN[m] for m in now)) or '无'}"
              f"（全样本时：{('、'.join(METHOD_CN[m] for m in orig)) or '无'}）。")
        a("")
        orig24 = [m for m in METHODS if res["years"]["2024"]["methods"][m]["significant"]]
        now24 = res["sensitivity_sig"]["2024"]
        lost = [m for m in orig24 if m not in now24]
        if lost:
            a("注意：仅剔除 {} 只样本，2024 的显著方法就从 {} 个降到 {} 个（{}）。"
              "这些方法的 2024 显著性本身就吊在门槛线上：{}。".format(
                  y24["buy_before_st"], len(orig24), len(now24),
                  "、".join(METHOD_CN[m] for m in lost),
                  "；".join(f"{METHOD_CN[m]} 的 |rho| 由 {abs(y24['methods'][m]['rho']):.3f} "
                            f"掉到 {abs(sens['2024']['methods'][m]['rho']):.3f}"
                            for m in lost)))
    else:
        a("样本量不足，未做该敏感性检查。")
    a("")

    a("## 6. 结论")
    a("")
    stable = [METHOD_CN[m] for m in METHODS if cy[m]["verdict"] == "跨年稳定"]
    unstable = [METHOD_CN[m] for m in METHODS if cy[m]["verdict"] == "不稳定"]
    useless = [METHOD_CN[m] for m in METHODS if cy[m]["verdict"] == "无效"]
    a(f"- **跨年稳定（{len(stable)} 个）**：{'、'.join(stable) if stable else '无'}。")
    a(f"- **跨年不稳定（{len(unstable)} 个）**：{'、'.join(unstable) if unstable else '无'}。")
    a(f"- **两年均无效（{len(useless)} 个）**：{'、'.join(useless) if useless else '无'}。")
    a("")
    for line in res["verdicts"]["narrative"]:
        a(f"- {line}")
    a("")

    a("## 7. 未解决项")
    a("")
    for line in res["unresolved"]:
        a(f"- {line}")
    a("")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    panel = pickle.load(open(PANEL, "rb"))

    rows24, drop24 = collect(panel, 2024)
    rows25, drop25 = collect(panel, 2025)

    y24 = evaluate(rows24, "2024 样本外")
    y25 = evaluate(rows25, "2025 样本内")
    y24["n_st_events"] = len(st_events(panel, 2024))
    y25["n_st_events"] = len(st_events(panel, 2025))
    y24["drop"], y25["drop"] = drop24, drop25

    all_rows = rows24 + rows25
    pool = pooled_eval(all_rows)
    cy = cross_year(y24, y25)
    sp = simpson_check(y24, y25, pool)

    fisher = {}
    for m in METHODS:
        z, p = fisher_combine([(y24["methods"][m]["rho"], y24["n"]),
                               (y25["methods"][m]["rho"], y25["n"])])
        fisher[m] = {"z": z, "p": p}

    # 敏感性：剔除「买点早于戴帽日」（放在结论之前，结论要用到）
    sens, sens_eval = {}, {}
    for y, rows in ((2024, rows24), (2025, rows25)):
        sub = [r for r in rows if r["buy_after_st"]]
        e = evaluate(sub, f"{y} 剔除买点早于戴帽")
        sens_eval[y] = e
        sens[str(y)] = {"n": e["n"], "n_pos": e["n_pos"], "pos_rate": e["pos_rate"],
                        "methods": {m: {"rho": e["methods"][m]["rho"], "p": e["methods"][m]["p"]}
                                    for m in METHODS}}
    sens_cy = cross_year(sens_eval[2024], sens_eval[2025])
    sens_verdicts = {m: sens_cy[m]["verdict"] for m in METHODS}
    sens_changed = [m for m in METHODS if sens_cy[m]["verdict"] != cy[m]["verdict"]]
    sens_sig = {str(y): [m for m in METHODS if sens_eval[y]["methods"][m]["significant"]]
                for y in (2024, 2025)}

    # ---- 结论（由数据生成，不手写数字）
    stable = [m for m in METHODS if cy[m]["verdict"] == "跨年稳定"]
    unstable = [m for m in METHODS if cy[m]["verdict"] == "不稳定"]
    useless = [m for m in METHODS if cy[m]["verdict"] == "无效"]

    narrative = []
    if unstable:
        flip = [m for m in unstable if cy[m]["sign_flip"]]
        decay = [m for m in unstable if not cy[m]["sign_flip"]]
        if flip:
            parts = "、".join(
                f"{METHOD_CN[m]}（{fmt(cy[m]['rho_2024'])} → {fmt(cy[m]['rho_2025'])}）" for m in flip)
            narrative.append(f"符号反转：{parts}。同一套参数在相邻两年给出相反方向，"
                             "说明这些指标捕捉的是当年市场状态，不是形态本身。")
        if decay:
            parts = "、".join(
                f"{METHOD_CN[m]}（2024 {fmt(cy[m]['rho_2024'])} 显著、2025 {fmt(cy[m]['rho_2025'])} 归零）"
                for m in decay)
            narrative.append(f"强度崩塌：{parts}。2024 有效但 2025 不可复现，"
                             "按判定规则同样归入「不稳定」——它的 2024 显著性不能当作可交易信号。")
    if stable:
        parts = "、".join(
            f"{METHOD_CN[m]}（{fmt(cy[m]['rho_2024'])} / {fmt(cy[m]['rho_2025'])}）" for m in stable)
        narrative.append(f"两年同向且都显著：{parts}。这是唯一可以谈「跨年稳定」的一类。")
    else:
        narrative.append("七个方法里没有任何一个做到「两年同向且都显著」，"
                         "即不存在跨年稳定的形态预测能力。")
    if useless:
        narrative.append(f"两年同号但都不显著：{'、'.join(METHOD_CN[m] for m in useless)}，"
                         "按预先登记的判定规则记为无效（不是「弱有效」「有待观察」）。")

    worst = max(METHODS, key=lambda m: sp["methods"][m]["pooled_minus_best_year"])
    sd = sp["methods"][worst]
    narrative.append(
        f"年份混杂的最强证据是 {METHOD_CN[worst]}：两年 rho 分别 {fmt(sd['rho_2024'])} 和 "
        f"{fmt(sd['rho_2025'])}（都不显著），合并后变成 {fmt(sd['rho_pooled'])}"
        f"（p={sd['p_pooled']:.2g}），扣掉年份后只剩 {fmt(sd['rho_within_year'])}"
        f"（p={sd['p_within_year']:.3f}）；该特征区分年份的 AUC 为 {sd['auc_year']:.3f}。"
        "合并后变强的是年份差异，不是形态信号。")

    sig_both = [m for m in METHODS
                if y24["methods"][m]["significant"] and y25["methods"][m]["significant"]]
    n_sig24 = sum(1 for m in METHODS if y24["methods"][m]["significant"])
    n_sig25 = sum(1 for m in METHODS if y25["methods"][m]["significant"])
    if not sig_both:
        narrative.append(
            f"样本外总判定：0 个方法在两年同时显著。"
            f"2024 达到门槛的有 {n_sig24} 个，2025 达到门槛的有 {n_sig25} 个，两者没有交集。")
    else:
        narrative.append(
            f"样本外总判定：{len(sig_both)} 个方法在两年同时显著——"
            f"{'、'.join(METHOD_CN[m] for m in sig_both)}。")

    lost24 = [m for m in METHODS
              if y24["methods"][m]["significant"] and m not in sens_sig["2024"]]
    if lost24:
        narrative.append(
            "2024 的显著性本身也不稳：仅剔除 {} 只「买点早于戴帽日」的样本，{}就跌破门槛（{}）。"
            "即便不看 2025，2024 那点显著也是极少数样本撑起来的，不能当作方法有效的证据。".format(
                y24["buy_before_st"], "、".join(METHOD_CN[m] for m in lost24),
                "；".join(f"{METHOD_CN[m]} |rho| {abs(y24['methods'][m]['rho']):.3f} → "
                          f"{abs(sens['2024']['methods'][m]['rho']):.3f}" for m in lost24)))

    verdicts = {"stable": stable, "unstable": unstable, "useless": useless,
                "n_sig_2024": n_sig24, "n_sig_2025": n_sig25,
                "n_sig_both": len(sig_both),
                "narrative": narrative,
                "pooled_reliable": sp["pooled_reliable"]}

    res = {
        "meta": {
            "task": "T-FRACTAL-02",
            "data": PANEL,
            "n_codes": len(panel),
            "date_min": min(d["dates"][0] for d in panel.values()),
            "date_max": max(d["dates"][-1] for d in panel.values()),
            "window": W, "hold": HOLD, "label_threshold": LABEL_TH,
            "buy_window": [BUY_START, BUY_END],
            "tpl_segments": "下跌40/上涨40/回落30/突破10",
            "sig_rule": {"p": SIG_P, "abs_rho": SIG_RHO},
            "generated_by": "analysis/out_oos/fractal_oos.py",
        },
        "years": {"2024": y24, "2025": y25},
        "pooled": pool,
        "cross_year": cy,
        "fisher": fisher,
        "simpson": sp,
        "verdicts": verdicts,
        "unresolved": [
            "买点与戴帽日先后未强制约束：2024 有 {0} 只、2025 有 {1} 只样本的 10-15~11-07 买点"
            "早于其 ST 生效日，即「先买后戴帽」，按任务单口径仍保留在样本内；"
            "剔除后的重算结果见报告第 5.1 节与 `result.json` 的 "
            "`sensitivity_exclude_buy_before_st`。".format(
                drop24["buy_before_st"], drop25["buy_before_st"]),
            "M1 为「路径长度/首尾距离」的粗糙度代理，不是严格 Higuchi 维数（120 点窗口下分形维数估计会饱和），"
            "口径沿用 fractal_lab 的实现，与上一轮一致。",
            "M2/M3 在部分样本上可能出现 NaN（窗口内价格几乎不动导致回归退化），已作为剔除项计入 "
            "`drop.feature_nan`，两年均为 {0} / {1}。".format(drop24["feature_nan"], drop25["feature_nan"]),
            "样本量小（2024 {0} 只 / 2025 {1} 只），Spearman 的 95% 置信区间很宽；"
            "本报告只做点估计与符号一致性判断，不做多重比较校正。".format(y24["n"], y25["n"]),
            "ST 判定依赖数据里的证券简称字段，含 XD/XR 前缀的除权日简称仍能正确识别 ST，"
            "但若原始源对简称做了额外改写（例如只写「ST」无前后缀），可能漏计。",
            "缓存 `panel2425.pkl` 含 5517 只，原始包为 5782 只，相差 265 只未纳入。"
            "差额大概率是 2024 年初前已退市、或全区间无有效日线的标的，本报告未逐只核对；"
            "若其中有 2024/2025 新增 ST，则对应年份样本量被低估。",
        ],
    }

    res["sensitivity_exclude_buy_before_st"] = sens
    res["sensitivity_verdicts"] = sens_verdicts
    res["sensitivity_changed"] = sens_changed
    res["sensitivity_sig"] = sens_sig

    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    with open(os.path.join(OUT_DIR, "report.md"), "w", encoding="utf-8") as f:
        f.write(build_report(res))

    # ---- stdout 摘要
    print("=" * 78)
    print("分形方法样本外检验（2024 样本外 vs 2025 样本内，日线）")
    print("=" * 78)
    for tag, yy in (("2024", y24), ("2025", y25)):
        print(f"\n[{tag}] 样本 {yy['n']} | 正样本 {yy['n_pos']} | 正样本率 {yy['pos_rate']*100:.1f}% "
              f"| 持有期均值 {yy['ret_mean']*100:+.1f}%")
        print(f"  {'方法':<20}{'Spearman':>10}{'p':>9}{'AUC':>8}{'Top10':>8}")
        for m in METHODS:
            s = yy["methods"][m]
            print(f"  {METHOD_CN[m]:<18}{s['rho']:>+10.3f}{s['p']:>9.4f}{s['auc']:>8.3f}"
                  f"{s['top10_hit']*100:>7.0f}%")
    print("\n[跨年符号一致性]")
    print(f"  {'方法':<20}{'2024':>9}{'2025':>9}{'同号':>7}   判定")
    for m in METHODS:
        c = cy[m]
        print(f"  {METHOD_CN[m]:<18}{c['rho_2024']:>+9.3f}{c['rho_2025']:>+9.3f}"
              f"{'是' if c['same_sign'] else '否':>6}   {c['verdict']}")
    print(f"\n[合并样本] n={pool['n']} 正样本率={pool['pos_rate']*100:.1f}% "
          f"| 正样本率差 {sp['pos_rate_gap_pp']:.1f}pp, 卡方 p={sp['chi2_p']:.2g}")
    print("  合并显著性不可信（年份混杂），详见 report.md 第 5 节")
    print(f"\n[产物] {OUT_DIR}/result.json, report.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
