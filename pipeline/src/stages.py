# -*- coding: utf-8 -*-
"""八阶段流水线。每阶段一个函数，输入 artifacts dict + spec，输出写盘 + 回填 artifacts。

阶段划分（对应 PIPELINE.md）：
  S0 规格校验   S1 数据接入   S2 单季化   S3 时点对齐
  S4 因子计算   S5 截面标准化 S6 评价     S7 报告
"""
import bisect
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (save_pkl, load_pkl, save_json, winsorize, cs_rank, neutralize,
                    spearman, nw_t, max_drawdown, annualize, sharpe, quantile_bins)

REQUIRED_SPEC = {
    "name": "因子名",
    "data.prices": "价格面板路径",
    "data.reports": "财报长表路径",
    "formula.type": "公式类型",
    "formula.window": "回归窗口（季度数）",
    "formula.min_n": "最小有效样本",
    "formula.ddof": "残差标准差自由度",
    "align.mode": "时点对齐方式",
    "universe.exclude_financial": "是否剔除金融",
    "preprocess.neutralize": "是否中性化",
    "evaluate.rebalance": "调仓间隔（交易日）",
    "evaluate.hold": "持有期（交易日）",
    "evaluate.cost_bps": "单边成本（bp）",
}


def _dig(d, path, default=None):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def default_spec():
    return {
        "name": "ocfs_demo",
        "data": {"prices": "runs/demo_data/prices.pkl",
                 "reports": "runs/demo_data/reports.pkl"},
        "formula": {"type": "ocfs_residual", "window": 8, "min_n": 6, "ddof": 3,
                    "scale": "none"},
        "align": {"mode": "ann_date"},
        "universe": {"exclude_financial": True, "exclude_st": True, "min_listed_days": 250},
        "preprocess": {"winsorize": True, "winsor_pct": [1, 99],
                       "rank": True, "neutralize": True},
        "evaluate": {"rebalance": 21, "hold": 21, "layers": 5, "cost_bps": 15},
    }


# ----------------------------- S0 规格校验 -----------------------------

def stage0_spec(spec, art, out_dir):
    missing = []
    for path, desc in REQUIRED_SPEC.items():
        v = _dig(spec, path, None)
        if v is None or (isinstance(v, str) and not v.strip()):
            missing.append(f"{path}（{desc}）")
    checks = {
        "missing_fields": missing,
        "align_mode": _dig(spec, "align.mode"),
        "min_n": _dig(spec, "formula.min_n"),
        "ddof": _dig(spec, "formula.ddof"),
        "neutralize": _dig(spec, "preprocess.neutralize"),
        "cost_bps": _dig(spec, "evaluate.cost_bps"),
    }
    save_json({"spec": spec, "checks": checks}, os.path.join(out_dir, "00_spec.json"))
    art["spec_checks"] = checks
    return checks


# ----------------------------- S1 数据接入 -----------------------------

def stage1_ingest(spec, art, out_dir):
    """载入数据并按 spec.universe 过滤股票池。过滤必须可核查：剔了谁、剔了多少、依据什么。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pp = _dig(spec, "data.prices")
    rp = _dig(spec, "data.reports")
    prices = load_pkl(pp if os.path.isabs(pp) else os.path.join(base, pp))
    reports = load_pkl(rp if os.path.isabs(rp) else os.path.join(base, rp))
    rows = reports["rows"]

    uni = _dig(spec, "universe", {}) or {}
    ex_fin = bool(uni.get("exclude_financial", False))
    ex_st = bool(uni.get("exclude_st", False))
    min_hist = int(uni.get("min_listed_days", 0) or 0)

    meta = prices["meta"]
    close = prices["close"]
    dropped = {"financial": 0, "st": 0, "short_history": 0}
    keep = []
    for c in prices["codes"]:
        m = meta.get(c, {})
        if ex_fin and m.get("is_financial"):
            dropped["financial"] += 1
            continue
        if ex_st and m.get("is_st"):
            dropped["st"] += 1
            continue
        if min_hist and len(close.get(c, [])) < min_hist:
            dropped["short_history"] += 1
            continue
        keep.append(c)
    kept = set(keep)

    n_before = len(prices["codes"])
    prices = {"dates": prices["dates"], "codes": keep,
              "close": {c: close[c] for c in keep if c in close},
              "meta": {c: meta[c] for c in keep if c in meta}}
    rows = [r for r in rows if r["code"] in kept]

    n_rep = len(rows)
    ann_dates = sorted({r["ann_date"] for r in rows})
    coverage = {
        "n_codes_price": len(prices["codes"]),
        "n_codes_before_filter": n_before,
        "n_codes_report": len({r["code"] for r in rows}),
        "n_reports": n_rep,
        "n_ann_dates": len(ann_dates),
        "price_range": [prices["dates"][0], prices["dates"][-1]],
        "report_range": [ann_dates[0], ann_dates[-1]] if ann_dates else [None, None],
        "null_cfo_share": round(sum(1 for r in rows if r.get("cfo_ytd") is None) / max(n_rep, 1), 4),
        "universe": {
            "applied": {"exclude_financial": ex_fin, "exclude_st": ex_st,
                        "min_listed_days": min_hist},
            "dropped": dropped,
            "note": "min_listed_days 用价格历史长度近似（数据无上市日字段）",
        },
    }
    art["prices"] = prices
    art["reports"] = rows
    art["coverage"] = coverage
    art["universe"] = coverage["universe"]
    save_json(coverage, os.path.join(out_dir, "01_coverage.json"))
    return coverage


# ----------------------------- S2 单季化 -----------------------------

def quarterize(rows):
    """
    输入长表（含 season 与累计值），输出 {code: [{period, ann_date, cfo, rcv}]}，按 period 升序。
    cfo = 单季值（累计差分，Q1 取原值）；rcv = 应收+票据+预付+存货。
    不做跨年差分：Q1 直接用当年累计。
    """
    by_code = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append(r)
    out = {}
    for code, rs in by_code.items():
        rs.sort(key=lambda x: x["period"])
        prev_year, prev_ytd = None, None
        series = []
        for r in rs:
            y = r["period"][:4]
            ytd = r.get("cfo_ytd")
            if ytd is None:
                cfo = None
            elif r["season"] == 1 or prev_year != y or prev_ytd is None:
                cfo = float(ytd)                      # Q1 或跨年首季：累计=单季
            else:
                cfo = float(ytd) - float(prev_ytd)
            prev_year, prev_ytd = y, ytd
            rcv = None
            if all(r.get(k) is not None for k in ("ar", "br", "ap", "inv")):
                rcv = float(r["ar"]) + float(r["br"]) + float(r["ap"]) + float(r["inv"])
            series.append({"period": r["period"], "season": r["season"],
                           "ann_date": r["ann_date"], "cfo": cfo, "rcv": rcv})
        # 计算 Δrcv（对相邻季），保留原始 rcv
        for i in range(len(series)):
            s = series[i]
            if s["rcv"] is None or i == 0 or series[i - 1]["rcv"] is None:
                s["d_rcv"] = None
            else:
                s["d_rcv"] = s["rcv"] - series[i - 1]["rcv"]
        out[code] = series
    return out


def stage2_quarterize(spec, art, out_dir):
    q = quarterize(art["reports"])
    art["quarterly"] = q
    n_series = sum(len(v) for v in q.values())
    n_cfo = sum(1 for v in q.values() for x in v if x["cfo"] is not None)
    n_drcv = sum(1 for v in q.values() for x in v if x["d_rcv"] is not None)
    rep = {"n_series": n_series, "cfo_valid": n_cfo, "d_rcv_valid": n_drcv,
           "cfo_valid_share": round(n_cfo / max(n_series, 1), 4),
           "d_rcv_valid_share": round(n_drcv / max(n_series, 1), 4),
           "de_cumulate": "Q1 取当年累计；其余 = 本期累计 - 上期累计；跨年重置"}
    save_pkl(q, os.path.join(out_dir, "02_quarterly.pkl"))
    save_json(rep, os.path.join(out_dir, "02_quarterly_report.json"))
    art["quarterly_report"] = rep
    return rep


# ----------------------------- S3 时点对齐 -----------------------------

def stage3_asof(spec, art, out_dir):
    """
    把每期数据挂到「公告日」上。若 spec.align.mode 不是 ann_date，则标记为前视风险。
    产出每只票的 [(ann_date, cfo, d_rcv)]，这是因子计算唯一允许使用的时点。
    """
    mode = _dig(spec, "align.mode")
    q = art["quarterly"]
    aligned = {}
    violations = 0
    for code, series in q.items():
        lst = []
        for s in series:
            fdate = s["ann_date"] if mode == "ann_date" else s["period"]
            if fdate < s["period"]:
                violations += 1
            lst.append({"date": fdate, "period": s["period"],
                        "cfo": s["cfo"], "d_rcv": s["d_rcv"]})
        lst.sort(key=lambda x: x["date"])
        aligned[code] = lst
    rep = {
        "mode": mode,
        "n_series": len(aligned),
        "n_points": sum(len(v) for v in aligned.values()),
        "violations_date_before_period": violations,
        "earliest_factor_date": min((v[0]["date"] for v in aligned.values() if v), default=None),
    }
    save_pkl(aligned, os.path.join(out_dir, "03_asof.pkl"))
    save_json(rep, os.path.join(out_dir, "03_asof_report.json"))
    art["aligned"] = aligned
    art["asof_report"] = rep
    return rep


# ----------------------------- S4 因子计算 -----------------------------

def _fit_predict(ys, x1s, x2s, ddof):
    """三参数 OLS：y = a + b*x1 + c*x2。返回 (残差, sigma, coefs) 或 None。"""
    y = np.asarray(ys, float)
    X = np.column_stack([np.ones(len(x1s)), np.asarray(x1s, float), np.asarray(x2s, float)])
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coefs
    if len(resid) - 3 <= 0:
        return None
    sigma = float(np.std(resid, ddof=ddof))
    return resid, sigma, coefs


def stage4_factor(spec, art, out_dir):
    """
    对每只票逐期做「一期向前」的残差预测：
      t 期因子 = (CFO_t - [a + b*CFO_{t-1} + c*Δrcv_t]) / sigma
      其中 a,b,c 只用 t 期之前的最多 window 期估计，sigma 同。
    这样 t 期因子不含 t 期以后的任何信息。
    """
    window = int(_dig(spec, "formula.window"))
    min_n = int(_dig(spec, "formula.min_n"))
    ddof = int(_dig(spec, "formula.ddof"))
    aligned = art["aligned"]
    factors = {}
    diag = {"n_clean": [], "beta": [], "gamma": [], "n_codes": 0, "n_points": 0,
            "n_skipped_short": 0}
    for code, ser in aligned.items():
        arr = [s for s in ser if s["cfo"] is not None and s["d_rcv"] is not None]
        if len(arr) < min_n + 2:
            diag["n_skipped_short"] += 1
            continue
        out = []
        for i in range(1, len(arr)):
            hist = arr[max(0, i - window):i]          # 只用 t 之前
            if len(hist) < min_n:
                diag["n_skipped_short"] += 1
                continue
            ys = [h["cfo"] for h in hist[1:]]
            x1s = [h["cfo"] for h in hist[:-1]]
            x2s = [h["d_rcv"] for h in hist[1:]]
            fit = _fit_predict(ys, x1s, x2s, ddof)
            if fit is None:
                continue
            _, sigma, coefs = fit
            if sigma <= 1e-12:
                continue
            cur = arr[i]
            pred = coefs[0] + coefs[1] * hist[-1]["cfo"] + coefs[2] * cur["d_rcv"]
            val = (cur["cfo"] - pred) / sigma
            out.append({"date": cur["date"], "value": float(val)})
            diag["n_clean"].append(len(hist))
            diag["beta"].append(float(coefs[1]))
            diag["gamma"].append(float(coefs[2]))
        if out:
            factors[code] = out
            diag["n_codes"] += 1
            diag["n_points"] += len(out)

    nc = np.asarray(diag["n_clean"], float)
    n_mean = float(np.mean(nc)) if len(nc) else float("nan")
    # σ̂ 的偏差：np.std(ddof=d) 除以 (n-d)，无偏需要 d=3（三个参数）
    inflation = float(np.sqrt(max(n_mean - ddof, 1e-9) / max(n_mean - 3, 1e-9))) if len(nc) else None
    rep = {
        "n_codes": diag["n_codes"], "n_points": diag["n_points"],
        "n_clean_median": float(np.median(nc)) if len(nc) else None,
        "n_clean_mean": n_mean if len(nc) else None,
        "n_clean_p10": float(np.percentile(nc, 10)) if len(nc) else None,
        "n_clean_lt_min_share": float(diag["n_skipped_short"] / max(diag["n_skipped_short"] + diag["n_points"], 1)),
        "beta_mean": float(np.mean(diag["beta"])) if diag["beta"] else None,
        "beta_sd": float(np.std(diag["beta"])) if diag["beta"] else None,
        "gamma_mean": float(np.mean(diag["gamma"])) if diag["gamma"] else None,
        "gamma_sign_pos_share": float(np.mean(np.array(diag["gamma"]) > 0)) if diag["gamma"] else None,
        "sigma_inflation": inflation,
        "window": window, "min_n": min_n, "ddof": ddof,
    }
    save_pkl(factors, os.path.join(out_dir, "04_factor_raw.pkl"))
    save_json(rep, os.path.join(out_dir, "04_factor_diag.json"))
    art["factor_raw"] = factors
    art["factor_diag"] = rep
    return rep


# ----------------------------- S5 截面标准化 -----------------------------

def stage5_standardize(spec, art, out_dir):
    """
    在统一的「月末交易日网格」上做截面处理。
    理由：因子按各自公告日生成，日期稀疏；截面操作（winsorize/中性化/排序）
    必须先落到同一日历上，否则每天只有七八只票，截面统计没有意义。
    """
    raw = art["factor_raw"]
    meta = art["prices"]["meta"]
    dates = art["prices"]["dates"]
    wpct = _dig(spec, "preprocess.winsor_pct", [1, 99])
    do_win = bool(_dig(spec, "preprocess.winsorize", True))
    do_rank = bool(_dig(spec, "preprocess.rank", True))
    do_neu = bool(_dig(spec, "preprocess.neutralize", False))

    fdates = {c: [x["date"] for x in v] for c, v in raw.items()}
    fvals = {c: [x["value"] for x in v] for c, v in raw.items()}

    # 调仓网格：每 step 个交易日，从第 250 个交易日起（step 来自 spec）
    step = int(_dig(spec, "evaluate.rebalance", 21) or 21)
    grid = list(range(250, len(dates) - 1, step))

    out = {}
    n_before, n_after, sizes = 0, 0, []
    for t in grid:
        d = dates[t]
        lst = []
        for c, dd in fdates.items():
            j = bisect.bisect_right(dd, d) - 1
            if j >= 0:
                lst.append((c, fvals[c][j]))
        if len(lst) < 50:
            continue
        n_before += len(lst)
        codes = [c for c, _ in lst]
        vals = np.array([v for _, v in lst], float)
        if do_win:
            vals = winsorize(vals, wpct[0], wpct[1])
        if do_neu:
            groups = [meta.get(c, {}).get("industry", "NA") for c in codes]
            sizes_mc = [meta.get(c, {}).get("mktcap", 1.0) for c in codes]
            vals = neutralize(vals, groups, sizes_mc)
        if do_rank:
            vals = cs_rank(vals)
        else:
            sd = np.std(vals)
            vals = (vals - vals.mean()) / sd if sd > 1e-12 else np.zeros_like(vals)
        out[d] = {c: float(v) for c, v in zip(codes, vals)}
        n_after += len(vals)
        sizes.append(len(vals))

    rep = {"n_dates": len(out), "n_before": n_before, "n_after": n_after,
           "grid": f"every_{step}d", "coverage_mean": float(np.mean(sizes)) if sizes else None,
           "winsorize": do_win, "rank": do_rank, "neutralize": do_neu,
           "neutralize_by": ["industry", "ln_mktcap"] if do_neu else []}
    save_pkl(out, os.path.join(out_dir, "05_factor_final.pkl"))
    save_json(rep, os.path.join(out_dir, "05_preprocess.json"))
    art["factor"] = out
    art["preprocess"] = rep
    return rep


# ----------------------------- S6 评价 -----------------------------

def stage6_evaluate(spec, art, out_dir):
    """在月末网格上做评价：IC / ICIR / NW-t / 衰减 / 分层 / 多空 / 换手 / 扣费。"""
    prices = art["prices"]
    dates = prices["dates"]
    close = prices["close"]
    factors = art["factor"]           # {date: {code: value}}
    hold = int(_dig(spec, "evaluate.hold"))
    k = int(_dig(spec, "evaluate.layers"))
    cost_bps = float(_dig(spec, "evaluate.cost_bps"))
    horizons = [1, 5, 10, 21, 63]

    idx_of = {d: i for i, d in enumerate(dates)}

    def fwd(code, t, h):
        cl = close.get(code)
        if not cl or t + h >= len(cl):
            return None
        a, b = cl[t], cl[t + h]
        return (b / a - 1.0) if a > 0 else None

    ic_list, used = [], []
    decay_acc = {h: [] for h in horizons}
    layer_acc = [[] for _ in range(k)]
    ls_list, nav, prev_hold, turns, sizes = [], [1.0], set(), [], []

    for d in sorted(factors):
        t = idx_of.get(d)
        if t is None or t + hold >= len(dates):
            continue
        cs = factors[d]
        codes = list(cs.keys())
        vals = np.asarray([cs[c] for c in codes], float)
        rets = np.asarray([fwd(c, t, hold) if fwd(c, t, hold) is not None else np.nan for c in codes])
        m = ~np.isnan(rets)
        if m.sum() < 50:
            continue
        ic = spearman(vals[m], rets[m])
        if not np.isnan(ic):
            ic_list.append(ic)
            used.append(d)
        for h in horizons:
            rr = np.asarray([fwd(c, t, h) if fwd(c, t, h) is not None else np.nan for c in codes])
            mm = ~np.isnan(rr)
            if mm.sum() >= 50:
                decay_acc[h].append(spearman(vals[mm], rr[mm]))
        b = quantile_bins(vals, k)
        for g in range(k):
            sel = (b == g) & m
            if sel.sum():
                layer_acc[g].append(float(np.mean(rets[sel])))
        top = set(np.asarray(codes)[b == k - 1])
        bot = set(np.asarray(codes)[b == 0])
        if top and bot:
            rl = [fwd(c, t, hold) for c in top]
            rs = [fwd(c, t, hold) for c in bot]
            rl = [x for x in rl if x is not None]
            rs = [x for x in rs if x is not None]
            if rl and rs:
                spread = float(np.mean(rl) - np.mean(rs))
                ls_list.append(spread)
                nav.append(nav[-1] * (1 + spread))
                cur = top | bot
                if prev_hold:
                    turns.append(len(cur ^ prev_hold) / max(len(cur), 1))
                prev_hold = cur
                sizes.append(len(codes))

    ic_arr = np.asarray(ic_list, float)
    # 月末网格 → 每年约 12 期
    gaps = [idx_of[b_] - idx_of[a_] for a_, b_ in zip(used[:-1], used[1:])]
    spacing = float(np.median(gaps)) if gaps else 21.0
    per_year = 252.0 / spacing
    ls = np.asarray(ls_list, float)
    turnover = float(np.mean(turns)) if turns else None
    cost_per_reb = (cost_bps / 1e4) * 2 * (turnover if turnover else 0)
    net_ls = ls - cost_per_reb if len(ls) else ls
    nw_lag = max(1, int(round(hold / spacing)))

    metrics = {
        "ic_mean": float(np.mean(ic_arr)) if len(ic_arr) else None,
        "ic_std": float(np.std(ic_arr, ddof=1)) if len(ic_arr) > 1 else None,
        "icir": (float(np.mean(ic_arr) / np.std(ic_arr, ddof=1))
                 if len(ic_arr) > 1 and np.std(ic_arr, ddof=1) > 0 else None),
        "ic_pos_ratio": float(np.mean(ic_arr > 0)) if len(ic_arr) else None,
        "nw_t": nw_t(ic_arr, nw_lag) if len(ic_arr) >= 5 else None,
        "nw_lag": nw_lag,
        "n_periods": int(len(ic_arr)),
        "n_independent_years": round(len(ic_arr) / per_year, 2),
        "ic_decay": {str(h): (float(np.nanmean(decay_acc[h])) if decay_acc[h] else None) for h in horizons},
        "layer_returns": [float(np.mean(x)) if x else None for x in layer_acc],
        "layer_monotonic": None,
        "ls_annual": annualize(ls, per_year) if len(ls) > 2 else None,
        "ls_sharpe": sharpe(ls, per_year) if len(ls) > 2 else None,
        "ls_mdd": max_drawdown(np.asarray(nav, float)) if len(nav) > 2 else None,
        "turnover": turnover,
        "cost_bps": cost_bps,
        "cost_per_rebalance": cost_per_reb,
        "net_ls_annual": annualize(net_ls, per_year) if len(net_ls) > 2 else None,
        "coverage_mean": float(np.mean(sizes)) if sizes else None,
        "coverage_min": int(np.min(sizes)) if sizes else None,
        "period_range": [used[0], used[-1]] if used else None,
        "hold": hold, "rebalance_days": spacing, "layers": k,
    }
    lr = [x for x in metrics["layer_returns"] if x is not None]
    if len(lr) == k and k >= 3:
        d1 = np.diff(lr)
        metrics["layer_monotonic"] = bool(np.all(d1 > 0) or np.all(d1 < 0))
    save_json(metrics, os.path.join(out_dir, "06_metrics.json"))
    art["metrics"] = metrics
    return metrics


# ----------------------------- S7 报告 -----------------------------

def stage7_report(spec, art, out_dir):
    m = art["metrics"]
    d = art["factor_diag"]
    p = art["preprocess"]
    c = art["coverage"]
    a = art["asof_report"]
    crit = spec.get("criteria") or {}
    ic_min = float(crit.get("ic_min", 0.02))
    t_min = float(crit.get("t_min", 2.0))

    def f(x, nd=3, pct=False):
        if x is None:
            return "未披露"
        return f"{x*100:.2f}%" if pct else f"{x:.{nd}f}"

    L = []
    L.append(f"# {spec.get('name','factor')} · 流水线报告\n")
    L.append("> 本报告所有数字由 `06_metrics.json` 程序化写入，无手填。\n")
    L.append("## 0. 规格\n")
    L.append(f"- 公式类型：`{_dig(spec,'formula.type')}`｜窗口 {d['window']} 季｜最小样本 {d['min_n']}｜σ 自由度 `ddof={d['ddof']}`")
    L.append(f"- 时点对齐：`{a['mode']}`｜预处理：winsorize={p['winsorize']} / rank={p['rank']} / 中性化={p['neutralize']}")
    L.append(f"- 调仓间距 {m['rebalance_days']:.0f} 交易日（等间隔网格）｜持有 {m['hold']} 日｜成本 {m['cost_bps']} bp 单边\n")
    L.append("## 1. 数据与覆盖\n")
    L.append(f"- 价格 {c['n_codes_price']} 只，{c['price_range'][0]} ~ {c['price_range'][1]}")
    L.append(f"- 财报 {c['n_reports']} 条，公告日 {c['n_ann_dates']} 个，`cfo` 缺失率 {c['null_cfo_share']*100:.2f}%")
    L.append(f"- 去累计口径：{art['quarterly_report']['de_cumulate']}\n")
    L.append("## 2. 时点检查\n")
    L.append(f"- 因子日期早于报告期（应为 0）：**{a['violations_date_before_period']}**")
    L.append(f"- 最早因子可用日：{a['earliest_factor_date']}\n")
    L.append("## 3. 估计诊断\n")
    L.append(f"- 有效样本中位 {f(d['n_clean_median'],1)} 季｜P10 {f(d['n_clean_p10'],1)}｜因历史不足被跳过占比 {f(d['n_clean_lt_min_share'],3,pct=True)}")
    L.append(f"- β 均值 {f(d['beta_mean'])}（σ {f(d['beta_sd'])}）｜γ 均值 {f(d['gamma_mean'])}｜γ>0 占比 {f(d['gamma_sign_pos_share'],3,pct=True)}")
    L.append(f"- σ 放大系数（√(n̄/(n̄−3))）：{f(d['sigma_inflation'])}（理想 ≈1.0，>1.2 说明分母被低估）\n")
    L.append("## 4. 因子表现\n")
    L.append("| 指标 | 数值 |\n|---|---|")
    rows = [("IC 均值", f(m['ic_mean'])), ("IC 标准差", f(m['ic_std'])),
            ("ICIR", f(m['icir'])), ("IC>0 占比", f(m['ic_pos_ratio'],3,pct=True)),
            ("NW-t（lag=调仓期）", f(m['nw_t'])), ("独立截面数", f(m['n_periods'],0)),
            ("等效独立年数", f(m['n_independent_years'])),
            ("多空年化", f(m['ls_annual'],3,pct=True)), ("多空夏普", f(m['ls_sharpe'])),
            ("多空最大回撤", f(m['ls_mdd'],3,pct=True)),
            ("换手率（单边）", f(m['turnover'],3,pct=True)),
            ("扣费后多空年化", f(m['net_ls_annual'],3,pct=True))]
    for k_, v_ in rows:
        L.append(f"| {k_} | {v_} |")
    L.append("\n### IC 衰减\n")
    L.append("| 持有期 | " + " | ".join(m['ic_decay'].keys()) + " |")
    L.append("|---" * (len(m['ic_decay']) + 1) + "|")
    L.append("| IC | " + " | ".join(f(v_) for v_ in m['ic_decay'].values()) + " |")
    L.append("\n### 分层收益（每期平均）\n")
    L.append("| 组 | " + " | ".join([f"L{i+1}" for i in range(m['layers'])]) + " |")
    L.append("|---" * (m['layers'] + 1) + "|")
    L.append("| 收益 | " + " | ".join(f(v_,3,pct=True) for v_ in m['layer_returns']) + " |")
    L.append(f"\n分层单调（程序判定）：**{m['layer_monotonic']}**\n")
    L.append("## 5. 结论（按预写判据自动判定）\n")
    verdict = []
    ic = m['ic_mean']
    t = m['nw_t']
    if ic is None:
        verdict.append(("IC", "无有效截面", "❓ 无法判定"))
    else:
        verdict.append(("方向", "IC>0" if ic > 0 else "IC≤0", "✅ 方向稳定" if ic > 0 else "❌ 方向为负"))
        verdict.append(("强度", f"IC={ic:.3f}", f"✅ 达到 {ic_min:.3f}" if ic >= ic_min else f"⚠️ 低于 {ic_min:.3f}"))
        if t is None:
            verdict.append(("显著性", "样本不足", "❓ 无法判定"))
        else:
            verdict.append(("显著性", f"NW-t={t:.2f}", f"✅ |t|>{t_min:g}" if abs(t) > t_min else "❌ 不显著"))
    verdict.append(("扣费后", "见上表", "✅ 为正" if (m['net_ls_annual'] or -1) > 0 else "❌ 扣费后不成立"))
    L.append("| 维度 | 读数 | 判定 |\n|---|---|---|")
    for a_, b_, c_ in verdict:
        L.append(f"| {a_} | {b_} | {c_} |")
    L.append("\n> 判据为**事前**写入，不随结果调整。合成数据仅用于验证流水线连通性，不可引用为真实市场结论。")

    path = os.path.join(out_dir, "07_report.md")
    with open(path, "w", encoding="utf-8") as f_:
        f_.write("\n".join(L))
    art["report_path"] = path
    return path
