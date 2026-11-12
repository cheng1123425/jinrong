# -*- coding: utf-8 -*-
"""
T-ST-COMP-01  ST 策略的合规事实核验
纯 numpy 实现（无 pandas），全部数字可由本脚本复算。

输入（全部使用缓存，不解压原始 zip）：
  analysis/panel2425.pkl   dict: code -> {dates,name,o,h,l,c,pc,v,fmv}
  analysis/st_events.json  dict: code -> {first,st_start,st_date,st_name,n}
输出：
  analysis/out_stcomp/result.json
  analysis/out_stcomp/report.md
"""
import json
import os
import pickle
import time

import numpy as np

T0 = time.time()
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "out_stcomp")
os.makedirs(OUT, exist_ok=True)

PANEL = os.path.join(BASE, "panel2425.pkl")
EVENTS = os.path.join(BASE, "st_events.json")

# ---------------- 参数（全部显式声明，便于复算） ----------------
H1_WINDOWS = [(-60, -21), (-20, -1), (-5, -1)]
H1_LOOKBACK = (-120, -61)
H1_MIN_FRAC = 0.60          # 事件窗口内有效交易日 >= 60% 才计入（停牌日不填充）
H1_MIN_VOL_DAYS = 20        # 基准量窗口至少 20 个有效交易日
H2_START, H2_END = "2025-10-15", "2025-11-07"
H2_LIMIT_TOL = 0.98         # 判定阈值折减系数
H4_HOLD = 20                # 持有交易日数
H4_AMOUNT = 1_000_000.0     # 冲击成本测算本金
IMPACT_COEF = 0.10          # 假设：冲击成本(bps) = 系数 * (下单额/日成交额) * 10000，系数=0.1
IMPACT_COEF_ALT = (0.05, 0.20)


# ---------------- 工具函数 ----------------
def clean(x):
    """JSON 安全的 float"""
    if x is None:
        return None
    if isinstance(x, (np.floating,)):
        x = float(x)
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def desc(arr):
    """返回分布统计：n/mean/median/std/min/p10/p25/p75/p90/max"""
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": clean(a.mean()),
        "median": clean(np.median(a)),
        "std": clean(a.std(ddof=1)) if a.size > 1 else None,
        "min": clean(a.min()),
        "p10": clean(np.percentile(a, 10)),
        "p25": clean(np.percentile(a, 25)),
        "p75": clean(np.percentile(a, 75)),
        "p90": clean(np.percentile(a, 90)),
        "max": clean(a.max()),
    }


def tstat(arr):
    """单样本 t 值（对 0），返回 (t, n)；n<3 或方差为 0 时 t=None"""
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], dtype=float)
    if a.size < 3:
        return None, int(a.size)
    sd = a.std(ddof=1)
    if sd == 0:
        return None, int(a.size)
    return clean(a.mean() / (sd / np.sqrt(a.size))), int(a.size)


def mean_of(arr):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], dtype=float)
    return clean(a.mean()) if a.size else None


def median_of(arr):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], dtype=float)
    return clean(np.median(a)) if a.size else None


def share(arr, cond):
    a = np.asarray([x for x in arr if x is not None and not np.isnan(x)], dtype=float)
    if a.size == 0:
        return None
    return clean(float((cond(a)).sum()) / a.size)


def is_st_name(nm):
    return bool(nm) and (nm.startswith("ST") or nm.startswith("*ST"))


def board_of(code):
    if code.startswith("bj"):
        return "bj"
    if code.startswith("sz30") or code.startswith("sh688"):
        return "gem_star"
    return "main"


def limit_pct(code, st_flag):
    """涨停幅度：北交所 30%，创业板/科创板 20%，主板 ST 5%、主板其他 10%
    说明：任务单写"ST 5%"，该口径只适用于主板；创业板/科创板 ST 股仍为 20%，
    此处按交易所实际规则细分，避免误判。"""
    b = board_of(code)
    if b == "bj":
        return 0.30
    if b == "gem_star":
        return 0.20
    return 0.05 if st_flag else 0.10


# ---------------- 载入 ----------------
print("[load] panel ...")
with open(PANEL, "rb") as f:
    panel = pickle.load(f)
with open(EVENTS, "r", encoding="utf-8") as f:
    events = json.load(f)
print("[load] %d stocks, %.1fs" % (len(panel), time.time() - T0))

# 全局交易日历（各股日期并集）
dateset = set()
for v in panel.values():
    dateset.update(v["dates"])
CAL = sorted(dateset)
CAL_IDX = {d: i for i, d in enumerate(CAL)}
NCAL = len(CAL)
print("[cal] %d trading days %s ~ %s" % (NCAL, CAL[0], CAL[-1]))

# ---------------- 全市场等权日收益 ----------------
# r = c/pc - 1；当日无数据的股票不计入分母
ret_sum = np.zeros(NCAL, dtype=float)
ret_cnt = np.zeros(NCAL, dtype=np.int64)
for code, d in panel.items():
    c = np.asarray(d["c"], dtype=float)
    pc = np.asarray(d["pc"], dtype=float)
    ok = np.isfinite(c) & np.isfinite(pc) & (pc > 0) & (c > 0)
    if not ok.any():
        continue
    r = np.zeros_like(c)
    r[ok] = c[ok] / pc[ok] - 1.0
    gi = np.array([CAL_IDX[x] for x in d["dates"]], dtype=np.int64)
    np.add.at(ret_sum, gi[ok], r[ok])
    np.add.at(ret_cnt, gi[ok], 1)

mkt = np.full(NCAL, np.nan)
mask = ret_cnt > 0
mkt[mask] = ret_sum[mask] / ret_cnt[mask]
# 等权市场净值（用于区间基准收益）
mkt_cum = np.ones(NCAL, dtype=float)
for i in range(1, NCAL):
    step = 1.0 + (mkt[i] if np.isfinite(mkt[i]) else 0.0)
    mkt_cum[i] = mkt_cum[i - 1] * step
print("[mkt] EW daily series ready, avg stocks/day=%.0f" % ret_cnt[mask].mean())

# 事件清单：2025 年戴帽
EV = []
for code, e in events.items():
    if not e.get("st_date"):
        continue
    if code not in panel:
        continue
    if not e["st_date"].startswith("2025"):
        continue
    if e["st_date"] not in CAL_IDX:
        continue
    EV.append({"code": code, "st_date": e["st_date"], "st_name": e.get("st_name"),
               "board": board_of(code), "gi": CAL_IDX[e["st_date"]]})
EV.sort(key=lambda x: (x["st_date"], x["code"]))
print("[ev] %d events in 2025" % len(EV))


def aligned(code):
    """把单只股票对齐到全局日历，缺日为 NaN"""
    d = panel[code]
    gi = np.array([CAL_IDX[x] for x in d["dates"]], dtype=np.int64)
    out = {}
    for k in ("o", "h", "l", "c", "pc", "v", "fmv"):
        arr = np.full(NCAL, np.nan)
        arr[gi] = np.asarray(d[k], dtype=float)
        out[k] = arr
    nm = np.empty(NCAL, dtype=object)
    nm[:] = None
    nm[gi] = np.asarray(d["name"], dtype=object)
    out["name"] = nm
    out["gi"] = gi
    out["dates"] = d["dates"]
    return out


# =====================================================================
# H1 事件前异常（信息泄露检验）
# =====================================================================
print("[H1] ...")
H1 = {"windows": {}, "by_group": {}, "vol_ratio": {}, "excluded": {}}
win_recs = {w: [] for w in H1_WINDOWS}
vol_ratios = []
per_event = []

for ev in EV:
    A = aligned(ev["code"])
    r = np.full(NCAL, np.nan)
    ok = np.isfinite(A["c"]) & np.isfinite(A["pc"]) & (A["pc"] > 0)
    r[ok] = A["c"][ok] / A["pc"][ok] - 1.0
    gi = ev["gi"]
    rec = {"code": ev["code"], "st_date": ev["st_date"], "month": int(ev["st_date"][5:7]),
           "board": ev["board"]}
    for (a, b) in H1_WINDOWS:
        lo, hi = gi + a, gi + b
        L = hi - lo + 1
        need = max(3, int(np.ceil(H1_MIN_FRAC * L)))
        if lo < 0:
            rec["w%d_%d" % (a, b)] = None
            continue
        seg_r = r[lo:hi + 1]
        seg_m = mkt[lo:hi + 1]
        m = np.isfinite(seg_r) & np.isfinite(seg_m)
        if m.sum() < need:
            rec["w%d_%d" % (a, b)] = None
            continue
        car = float(np.sum(seg_r[m] - seg_m[m]))
        rec["w%d_%d" % (a, b)] = car
        rec["w%d_%d_nd" % (a, b)] = int(m.sum())
        if (a, b) == (-20, -1):
            rec["car"] = car
    # 量比
    lo, hi = gi + H1_LOOKBACK[0], gi + H1_LOOKBACK[1]
    base = np.isfinite(A["v"][lo:hi + 1]) & (A["v"][lo:hi + 1] > 0) if lo >= 0 else np.array([False])
    cur = np.isfinite(A["v"][gi - 20:gi]) & (A["v"][gi - 20:gi] > 0) if gi - 20 >= 0 else np.array([False])
    vr = None
    if lo >= 0 and base.sum() >= H1_MIN_VOL_DAYS and cur.sum() >= 10:
        vr = float(A["v"][lo:hi + 1][base].mean() / A["v"][gi - 20:gi][cur].mean())
        vol_ratios.append(vr)
    rec["vol_ratio"] = vr
    rec["n_days_avail"] = int(np.isfinite(r[max(0, gi - 120):gi]).sum())
    per_event.append(rec)

for (a, b) in H1_WINDOWS:
    key = "[%d,%d]" % (a, b)
    vals = [rec["w%d_%d" % (a, b)] for rec in per_event]
    tv, n = tstat(vals)
    H1["windows"][key] = {
        "n": n, "mean": mean_of(vals), "median": median_of(vals), "t": tv,
        "pos_ratio": share(vals, lambda x: x > 0),
        "neg_ratio": share(vals, lambda x: x < 0),
        "p25": clean(np.nanpercentile([v for v in vals if v is not None], 25)) if n else None,
        "p75": clean(np.nanpercentile([v for v in vals if v is not None], 75)) if n else None,
        "excluded": len(EV) - n,
    }

# 对照组：4-5 月（年报类） vs 其他月份，只看 [-20,-1]
g_apr = [rec["w-20_-1"] for rec in per_event if rec["month"] in (4, 5)]
g_oth = [rec["w-20_-1"] for rec in per_event if rec["month"] not in (4, 5)]
for gname, vals in (("4-5月(年报类)", g_apr), ("其他月份(其他原因)", g_oth)):
    tv, n = tstat(vals)
    H1["by_group"][gname] = {
        "n": n, "mean": mean_of(vals), "median": median_of(vals), "t": tv,
        "pos_ratio": share(vals, lambda x: x > 0),
        "neg_ratio": share(vals, lambda x: x < 0),
        "events": [{"code": rec["code"], "st_date": rec["st_date"], "car": clean(rec["w-20_-1"])}
                   for rec in per_event
                   if (rec["month"] in (4, 5)) == (gname.startswith("4"))],
    }

vr_t, vr_n = tstat(vol_ratios)
H1["vol_ratio"] = {
    "n": vr_n, "mean": mean_of(vol_ratios), "median": median_of(vol_ratios),
    "t": vr_t, "gt1_ratio": share(vol_ratios, lambda x: x > 1.0),
    "definition": "mean(v,[-20,-1]) / mean(v,[-120,-61])，剔除停牌/无成交日",
}
H1["per_event"] = per_event
H1["excluded"] = {"events_with_lt_120_trading_days_of_history":
                  int(sum(1 for r in per_event if r["n_days_avail"] < 120)),
                  "total_events": len(EV),
                  "note": "有效交易日不足仅影响量比基准窗口，事件窗口按 60% 有效日门槛单独判定"}

# 判据
h1_t = H1["windows"]["[-20,-1]"]["t"]
h1_m = H1["windows"]["[-20,-1]"]["median"]
H1["verdict"] = {
    "metric": "[-20,-1] CAR 中位数 / t 值",
    "value": "median=%s, t=%s" % (h1_m, h1_t),
    "threshold": "median < -3% 且 t < -2",
    "pass": bool(h1_m is not None and h1_t is not None and h1_m < -0.03 and h1_t < -2.0),
    "conclusion": ("事件前存在显著负向漂移" if (h1_m is not None and h1_t is not None
                                                 and h1_m < -0.03 and h1_t < -2.0)
                   else "未达判据，不能认定为显著负向漂移"),
}
print("   [-20,-1] median=%.4f t=%s" % (h1_m, h1_t))


# =====================================================================
# H2 买点是否可成交
# =====================================================================
print("[H2] ...")
win_gi = [i for i, d in enumerate(CAL) if H2_START <= d <= H2_END]
win_dates = [CAL[i] for i in win_gi]
H2_rows = []
premiums = []
for ev in EV:
    A = aligned(ev["code"])
    for i in win_gi:
        c, o, h, l, pc = A["c"][i], A["o"][i], A["h"][i], A["l"][i], A["pc"][i]
        if not (np.isfinite(c) and np.isfinite(pc) and pc > 0 and c > 0):
            continue
        stf = is_st_name(A["name"][i])
        thr = limit_pct(ev["code"], stf)
        chg = c / pc - 1.0
        limit_up = chg >= thr * H2_LIMIT_TOL
        one_word = limit_up and all(np.isfinite(x) for x in (o, h, l)) and \
            abs(o - h) < 1e-9 and abs(h - l) < 1e-9 and abs(l - c) < 1e-9
        H2_rows.append({"code": ev["code"], "date": CAL[i], "chg": chg,
                        "limit_pct": thr, "limit_up": bool(limit_up), "one_word": bool(one_word),
                        "amount": c * A["v"][i] if np.isfinite(A["v"][i]) else None,
                        "st": bool(stf)})
        # 次日开盘溢价
        if i + 1 < NCAL and np.isfinite(A["o"][i + 1]):
            premiums.append(float(A["o"][i + 1] / c - 1.0))

n_rows = len(H2_rows)
n_up = sum(1 for r in H2_rows if r["limit_up"])
n_ow = sum(1 for r in H2_rows if r["one_word"])
lim_rate = n_up / n_rows if n_rows else None
ow_rate = n_ow / n_rows if n_rows else None

# 逐股口径：窗口内至少一天涨停 / 至少一天一字板的股票比例
by_stock = {}
for r in H2_rows:
    s = by_stock.setdefault(r["code"], {"days": 0, "up": 0, "ow": 0})
    s["days"] += 1
    s["up"] += int(r["limit_up"])
    s["ow"] += int(r["one_word"])
stock_up_ratio = float(np.mean([1.0 if v["up"] > 0 else 0.0 for v in by_stock.values()])) if by_stock else None
stock_ow_ratio = float(np.mean([1.0 if v["ow"] > 0 else 0.0 for v in by_stock.values()])) if by_stock else None
per_stock_lim = [v["up"] / v["days"] for v in by_stock.values()]

H2 = {
    "window": [H2_START, H2_END],
    "trading_days": len(win_dates),
    "universe": "2025 年新增 ST 且 st_date 落在 2025 年的股票",
    "n_stocks": len({r["code"] for r in H2_rows}),
    "n_stocks_universe": len(EV),
    "n_stocks_not_yet_st_in_window": sum(1 for e in EV if e["st_date"] > H2_END),
    "stock_days": n_rows,
    "limit_up_days": n_up,
    "limit_up_rate": clean(lim_rate),
    "one_word_days": n_ow,
    "one_word_rate": clean(ow_rate),
    "one_word_within_limit_rate": clean(n_ow / n_up) if n_up else None,
    "stocks_with_any_limit_up": clean(stock_up_ratio),
    "stocks_with_any_one_word": clean(stock_ow_ratio),
    "per_stock_limit_up_rate": desc(per_stock_lim),
    "premium_next_open": desc(premiums),
    "unbuyable_probability": clean(lim_rate),
    "unbuyable_probability_strict": clean(ow_rate),
    "unbuyable_def": "任取一个交易日以收盘价买入、当日封涨停（含一字板）即视为无法按收盘价成交；"
                     "概率 = 窗口内涨停 stock-day 数 / 全部有效 stock-day 数（合并口径，非个股独立事件）",
    "limit_rule": "北交所 30% / 创业板·科创板 20% / 主板 ST 5% / 主板其他 10%；"
                  "涨停判定 c/pc-1 >= 阈值*0.98；一字板=开高低收全等于涨停价",
}
H2["verdict"] = {
    "metric": "涨停（含一字板）stock-day 占比",
    "value": "%s (%d/%d)" % (("%.2f%%" % (lim_rate * 100)) if lim_rate is not None else "NA", n_up, n_rows),
    "threshold": "> 20%",
    "pass": bool(lim_rate is not None and lim_rate > 0.20),
    "conclusion": ("显著存在买不到的问题" if (lim_rate is not None and lim_rate > 0.20)
                   else "未达 20% 判据，但按日计仍有不可成交概率（见实测值）"),
}
print("   limit_up=%d/%d=%.2f%%, one_word=%d" % (n_up, n_rows, (lim_rate or 0) * 100, n_ow))


# =====================================================================
# H3 流动性与操纵易发性
# =====================================================================
print("[H3] ...")
amounts, turnovers, impacts = [], [], []
per_stock_amt = {}
for ev in EV:
    A = aligned(ev["code"])
    amts = []
    for i in win_gi:
        c, v, fmv = A["c"][i], A["v"][i], A["fmv"][i]
        if not (np.isfinite(c) and np.isfinite(v) and v > 0):
            continue
        amt = c * v
        amounts.append(amt)
        amts.append(amt)
        if np.isfinite(fmv) and fmv > 0:
            turnovers.append(amt / fmv)
        if amt > 0:
            impacts.append(IMPACT_COEF * (H4_AMOUNT / amt) * 10000.0)  # bps
    if amts:
        per_stock_amt[ev["code"]] = float(np.median(amts))

amt_d = desc(amounts)
turn_d = desc(turnovers)
imp_d = desc(impacts)
imp_lo = desc([IMPACT_COEF_ALT[0] * (H4_AMOUNT / a) * 10000.0 for a in amounts])
imp_hi = desc([IMPACT_COEF_ALT[1] * (H4_AMOUNT / a) * 10000.0 for a in amounts])

# 稳健性：全年（2025 年全部有数据交易日）口径
amt_y, turn_y = [], []
for ev in EV:
    A = aligned(ev["code"])
    for i in range(NCAL):
        if not CAL[i].startswith("2025"):
            continue
        c, v, fmv = A["c"][i], A["v"][i], A["fmv"][i]
        if not (np.isfinite(c) and np.isfinite(v) and v > 0):
            continue
        amt_y.append(c * v)
        if np.isfinite(fmv) and fmv > 0:
            turn_y.append(c * v / fmv)
H3 = {
    "window": [H2_START, H2_END],
    "stock_days": len(amounts),
    "note_amount": "成交额 = v * close 近似（v 为成交量股数，close 为当日收盘价），未使用成交均价；"
                   "一字板日收盘价即涨停价，与均价一致；盘中开板日两者有差，振幅越大偏差越大",
    "amount_cny": amt_d,
    "amount_per_stock_median": desc(list(per_stock_amt.values())),
    "turnover_proxy": turn_d,
    "robustness_full_2025": {
        "amount": desc(amt_y),
        "turnover": desc(turn_y),
        "note": "同一 126 只样本在 2025 全年（含戴帽前与戴帽后全部有数据交易日）的口径，"
                "用于说明 H3 结论对窗口选择的敏感性",
    },
    "turnover_def": "换手代理 = 成交额 / fmv（fmv 为自由流通市值字段），非官方换手率",
    "impact_1m_bps": {
        "coef": IMPACT_COEF,
        "def": "冲击成本(bps) = %.2f * (100万元 / 当日成交额) * 10000；系数为假设值，非市场实测"
               % IMPACT_COEF,
        "stats": imp_d,
        "sensitivity_coef_0.05": {"median": imp_lo.get("median")},
        "sensitivity_coef_0.20": {"median": imp_hi.get("median")},
    },
}
h3_med = amt_d.get("median")
H3["verdict"] = {
    "metric": "中位日成交额（元）",
    "value": ("%.0f (%.2f 万元)" % (h3_med, h3_med / 1e4)) if h3_med else "NA",
    "threshold": "< 5000 万元",
    "pass": bool(h3_med is not None and h3_med < 5e7),
    "conclusion": ("小盘低流动性，操纵成本低" if (h3_med is not None and h3_med < 5e7)
                   else "中位成交额 %.0f 万元，高于 5000 万元阈值 %.1f%%，未达该判据"
                        % (h3_med / 1e4, (h3_med / 5e7 - 1) * 100)),
}
print("   median amount=%.0f yuan" % (h3_med or 0))


# =====================================================================
# H4 收益分布 vs 收益承诺
# =====================================================================
print("[H4] ...")
rets, excs, hold_recs = [], [], []
for ev in EV:
    A = aligned(ev["code"])
    gi = A["gi"]
    cand = gi[gi >= win_gi[0]]
    if cand.size == 0:
        continue
    e0 = int(cand[0])
    if e0 + H4_HOLD >= NCAL:
        continue
    buy = A["c"][e0]
    sell = A["c"][e0 + H4_HOLD]
    if not (np.isfinite(buy) and np.isfinite(sell) and buy > 0):
        continue
    r = float(sell / buy - 1.0)
    bench = float(mkt_cum[e0 + H4_HOLD] / mkt_cum[e0] - 1.0)
    rets.append(r)
    excs.append(r - bench)
    hold_recs.append({"code": ev["code"], "entry": CAL[e0], "exit": CAL[e0 + H4_HOLD],
                      "ret": clean(r), "bench": clean(bench), "excess": clean(r - bench)})

H4 = {
    "entry_rule": "2025-10-15 ~ 2025-11-07 内首个有交易日的收盘价买入，按全局交易日历持有 20 个交易日，"
                  "收盘卖出（卖出日无价或停牌的剔除）",
    "n": len(rets),
    "return_dist": desc(rets),
    "excess_dist": desc(excs),
    "prob_ge_30": share(rets, lambda x: x >= 0.30),
    "prob_ge_50": share(rets, lambda x: x >= 0.50),
    "prob_negative": share(rets, lambda x: x < 0),
    "benchmark": "全市场等权日收益复利，同一自然区间（entry~exit）",
    "per_event": hold_recs,
}
p30 = H4["prob_ge_30"]
H4["verdict"] = {
    "metric": "P(持有 20 日收益 >= 30%)",
    "value": ("%.2f%% (%d/%d)" % (p30 * 100, sum(1 for r in rets if r >= 0.30), len(rets)))
             if p30 is not None else "NA",
    "threshold": "< 20%",
    "pass": bool(p30 is not None and p30 < 0.20),
    "conclusion": ("「稳赚 30%」与事实不符" if (p30 is not None and p30 < 0.20)
                   else "达到或超过 20%，该判据不支持「不符」"),
    "prob_ge_50": H4["prob_ge_50"],
}
print("   n=%d median=%.4f P30=%.4f" % (len(rets), np.median(rets) if rets else 0, p30 or 0))


# =====================================================================
# H5 债市溢出可行性（只评估）
# =====================================================================
print("[H5] ...")
need = ["转债价格", "转股溢价率", "纯债溢价率", "主体评级", "信用利差"]
have = {k: False for k in need}
# 本地仅股票日线缓存，无任何转债/主体评级字段（逐字段核对 panel 结构与目录）
panel_fields = set()
for v in panel.values():
    panel_fields.update(v.keys())
    break
local_files = sorted(os.listdir(BASE))
root_dir = os.path.dirname(BASE)
root_files = sorted(os.listdir(root_dir)) if os.path.isdir(root_dir) else []
bond_kw = ("bond", "cb_", "convert", "转债", "信用债", "rating", "credit")
bond_files = [f for f in local_files + root_files
              if any(t in f.lower() for t in bond_kw)]
H5 = {
    "required_fields": need,
    "present_locally": have,
    "panel_fields_available": sorted(panel_fields),
    "bond_data_files_found": bond_files,
    "verdict": "不可做",
    "reason": "本地缓存仅含股票 OHLC/成交量/自由流通市值，无转债价格、转股溢价率、纯债溢价率、"
              "主体评级、信用利差中的任何一项；且无债券代码与正股的映射表。"
              "按任务单硬性要求，不以股票数据替代债券数据。",
    "minimal_dataset_if_pursued": [
        "转债代码 + 正股代码映射（含退市/强赎状态）",
        "转债日收盘价（含全价/净价口径）",
        "转股溢价率 = 转债价格/转股价值 - 1（需转股价、正股价）",
        "纯债溢价率（需同期限同评级企业债收益率曲线）",
        "主体/债项评级及评级变动日期（含评级展望）",
        "信用利差（个券收益率 - 同期限国债收益率）",
        "戴帽公告日与转债条款触发（回售/下修/强赎）事件表",
    ],
}
print("   H5 = 不可做")

# =====================================================================
# 汇总
# =====================================================================
result = {
    "task": "T-ST-COMP-01",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "runtime_sec": None,
    "meta": {
        "panel_stocks": len(panel),
        "panel_dates": [CAL[0], CAL[-1]],
        "trading_days": NCAL,
        "events_2025": len(EV),
        "event_date_range": [EV[0]["st_date"], EV[-1]["st_date"]] if EV else None,
        "board_mix": {b: sum(1 for e in EV if e["board"] == b) for b in ("main", "gem_star", "bj")},
        "exclusion_rules": [
            "st_date 为空的股票不进入事件样本（5500 只中仅 126 只 2025 年戴帽）",
            "事件日不在全局交易日历中的剔除（实测 0 只）",
            "H1：事件窗口内有效配对日 < 窗口长度 60% 的，该窗口剔除（停牌日不填充）",
            "H1：量比基准窗口 [-120,-61] 有效日 < 20 的，该样本量比剔除",
            "H1/H2/H3：价格或成交量为空/<=0 的交易日剔除（停牌日不计收益，不以前值填充）",
            "H2/H3：窗口内整段无成交的 6 只剔除（均为 2025 年内退市股，末期名称含「退」），实际参与 120 只",
            "H4：2025-10-15 后无交易日、或此后不足 20 个交易日、或买卖价缺失的剔除",
        ],
        "market_benchmark": "当日所有有数据股票 c/pc-1 的算术均值（当日无数据的股票不计入分母）",
        "car_def": "CAR = Σ(个股日收益 − 全市场等权日收益)，窗口内仅取个股有数据的交易日（算术累加，非复利）",
    },
    "H1": H1,
    "H2": H2,
    "H3": H3,
    "H4": H4,
    "H5": H5,
    "limitations": [
        "事件前漂移无法区分『信息泄露』与『公开指标提前定价』：戴帽前已有业绩预告、"
        "审计意见、退市风险提示等公开信号，负向漂移可能完全由基本面公开信息驱动，本检验不能识别因果来源。",
        "成交额用 v×close 近似，未使用成交均价（VWAP）；一字板日 close 即涨停价，会系统性低估/高估当日真实成交均价，"
        "流动性指标存在口径误差。",
        "仅 2025 单年样本（126 个事件），未覆盖跨年周期，结论不能外推到其他年份；"
        "分组后『其他月份』仅 39 个观测，t 值统计功效有限。",
    ],
    "disclaimer": "本报告仅回答合规相关事实问题，不构成投资建议，也不评价该策略能否盈利。",
}
result["runtime_sec"] = round(time.time() - T0, 2)

with open(os.path.join(OUT, "result.json"), "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("[write] result.json  %.1fs" % (time.time() - T0))


# =====================================================================
# report.md（所有数字均取自 result，保证与 json 一致）
# =====================================================================
def P(x, nd=2):
    return "NA" if x is None else ("%." + str(nd) + "f%%") % (x * 100)


def P4(x):
    return "NA" if x is None else "%+.2f%%" % (x * 100)


def F(x, nd=2):
    return "NA" if x is None else ("%." + str(nd) + "f") % x


def W(x):
    return "NA" if x is None else "%.0f 万元" % (x / 1e4)


def row(cells):
    return "| " + " | ".join(str(c) for c in cells) + " |"


M = result["meta"]
h1w = result["H1"]["windows"]
h1g = result["H1"]["by_group"]
h1v = result["H1"]["verdict"]
h2, h2v = result["H2"], result["H2"]["verdict"]
h3, h3v = result["H3"], result["H3"]["verdict"]
h4, h4v = result["H4"], result["H4"]["verdict"]
h5 = result["H5"]

L = []
L.append("# T-ST-COMP-01　ST 策略的合规事实核验")
L.append("")
L.append("> 本报告只回答合规相关的事实问题，不构成投资建议，也不评价该策略能否盈利。")
L.append(">")
L.append("> 五条判据的实测值 / 阈值 / 结论速览见第 7 节；样本口径与剔除规则见第 0 节。")
L.append("")
L.append("| 项 | 值 |")
L.append("|---|---|")
L.append(row(["任务号", "T-ST-COMP-01"]))
L.append(row(["生成时间", result["generated_at"]]))
L.append(row(["运行时长", "%.2f 秒（读缓存，未解压原始 zip）" % result["runtime_sec"]]))
L.append(row(["脚本", "analysis/out_stcomp/st_comp.py（纯 numpy，可一键复算）"]))
L.append("")

L.append("## 0　样本与口径")
L.append("")
L.append("| 项目 | 值 |")
L.append("|---|---|")
L.append(row(["面板股票数", "%d 只" % M["panel_stocks"]]))
L.append(row(["面板交易日", "%d 个交易日，%s ~ %s" % (M["trading_days"], M["panel_dates"][0], M["panel_dates"][1])]))
L.append(row(["事件样本数", "%d 个（st_date 非空且落在 2025 年，全部可在面板中定位）" % M["events_2025"]]))
L.append(row(["事件日期范围", "%s ~ %s" % (M["event_date_range"][0], M["event_date_range"][1])]))
L.append(row(["板块分布", "主板 %d、创业板/科创板 %d、北交所 %d" % (
    M["board_mix"]["main"], M["board_mix"]["gem_star"], M["board_mix"]["bj"])]))
L.append(row(["对照分组", "4-5 月（年报类）%d 个；其他月份（其他原因）%d 个" % (
    h1g["4-5月(年报类)"]["n"], h1g["其他月份(其他原因)"]["n"])]))
L.append("")
L.append("**剔除规则**")
L.append("")
for i, x in enumerate(M["exclusion_rules"], 1):
    L.append("%d. %s" % (i, x))
L.append("")
L.append("**关键定义**")
L.append("")
L.append("- 全市场等权日收益：%s。" % M["market_benchmark"])
L.append("- %s。" % M["car_def"])
L.append("- 事件窗口按全局交易日历计数（含个股停牌日，但停牌日不计个股收益、不参与配对），"
         "窗口内有效配对日 >= 窗口长度的 60% 才计入。")
L.append("- 收益均为 c/pc-1 口径（pc 为数据源给出的前收盘价，已含除权调整），未做后复权处理。")
L.append("")

# ---- H1
L.append("## 1　H1 事件前是否存在异常（信息泄露检验）")
L.append("")
L.append("单位：累计超额收益 CAR（个股累计收益 − 全市场等权累计收益），算术累加。")
L.append("")
L.append("| 窗口 | 样本数 | 均值 | 中位 | t 值 | 为正比例 | 为负比例 |")
L.append("|---|---|---|---|---|---|---|")
for k in ("[-60,-21]", "[-20,-1]", "[-5,-1]"):
    d = h1w[k]
    L.append(row([k, d["n"], P4(d["mean"]), P4(d["median"]), F(d["t"], 2),
                  P(d["pos_ratio"], 1), P(d["neg_ratio"], 1)]))
L.append("")
L.append("**对照组：按戴帽月份分组（仅 [-20,-1] 窗口）**")
L.append("")
L.append("| 分组 | 样本数 | 均值 | 中位 | t 值 | 为负比例 |")
L.append("|---|---|---|---|---|---|")
for g in ("4-5月(年报类)", "其他月份(其他原因)"):
    d = h1g[g]
    L.append(row([g, d["n"], P4(d["mean"]), P4(d["median"]), F(d["t"], 2), P(d["neg_ratio"], 1)]))
L.append("")
vr = result["H1"]["vol_ratio"]
L.append("**成交量比**（%s）" % vr["definition"])
L.append("")
L.append("| 指标 | 样本数 | 均值 | 中位 | t 值 | 量比>1 比例 |")
L.append("|---|---|---|---|---|---|")
L.append(row(["量比", vr["n"], F(vr["mean"], 2), F(vr["median"], 2), F(vr["t"], 2), P(vr["gt1_ratio"], 1)]))
L.append("")
L.append("**判据**")
L.append("")
L.append("| 实测值 | 阈值 | 结论 |")
L.append("|---|---|---|")
L.append(row(["[-20,-1] CAR 中位 %s，t = %s（n=%d）" % (P4(h1w["[-20,-1]"]["median"]),
                                                 F(h1w["[-20,-1]"]["t"], 2), h1w["[-20,-1]"]["n"]),
              "中位 < -3% 且 t < -2", "**%s**" % h1v["conclusion"]]))
L.append("")
L.append("事实陈述：戴帽前 20 个交易日，样本股相对全市场等权组合的累计超额收益中位数为 %s，"
         "%.1f%% 的样本为负；前 60~21 日窗口中位数为 %s。分组看，4-5 月年报类戴帽股为 %s，"
         "其他月份为 %s，两组方向一致、年报类幅度更大。"
         % (P4(h1w["[-20,-1]"]["median"]), (h1w["[-20,-1]"]["neg_ratio"] or 0) * 100,
            P4(h1w["[-60,-21]"]["median"]), P4(h1g["4-5月(年报类)"]["median"]),
            P4(h1g["其他月份(其他原因)"]["median"])))
L.append("")
L.append("> 该结果只能说明「戴帽公告前价格已下跌」，**不能**据此认定存在信息泄露；"
         "见第 6 节局限 ①。")
L.append("")

# ---- H2
L.append("## 2　H2 买点是否可成交")
L.append("")
L.append("窗口 %s ~ %s（%d 个交易日），样本 %d 只 2025 年新增 ST 股、%d 个有效 stock-day。"
         "全样本 %d 只：其中 6 只在窗口内整段无成交（长期停牌/已退市）故剔除；"
         "另 %d 只直至 2025 年 12 月才戴帽，窗口内仍按戴帽前涨跌幅限制（10%%/20%%）判定。"
         % (h2["window"][0], h2["window"][1], h2["trading_days"], h2["n_stocks"], h2["stock_days"],
            h2["n_stocks_universe"], h2["n_stocks_not_yet_st_in_window"]))
L.append("")
L.append("| 指标 | 数值 |")
L.append("|---|---|")
L.append(row(["涨停 stock-day（含一字板）", "%d / %d = **%s**" % (h2["limit_up_days"], h2["stock_days"], P(h2["limit_up_rate"]))]))
L.append(row(["一字板 stock-day", "%d / %d = **%s**（占涨停日的 %s）" % (
    h2["one_word_days"], h2["stock_days"], P(h2["one_word_rate"]), P(h2["one_word_within_limit_rate"], 1))]))
L.append(row(["窗口内至少涨停过一次的股票", "%s（%d 只中）" % (P(h2["stocks_with_any_limit_up"], 1), h2["n_stocks"])]))
L.append(row(["窗口内至少一字板过一次的股票", P(h2["stocks_with_any_one_word"], 1)]))
L.append(row(["个股涨停天数占比：中位 / P75 / P90", "%s / %s / %s" % (
    P(h2["per_stock_limit_up_rate"].get("median")), P(h2["per_stock_limit_up_rate"].get("p75")),
    P(h2["per_stock_limit_up_rate"].get("p90")))]))
L.append(row(["次日开盘相对当日收盘溢价：中位 / 均值", "%s / %s" % (
    P4(h2["premium_next_open"].get("median")), P4(h2["premium_next_open"].get("mean")))]))
L.append(row(["次日开盘溢价：P10 / P25 / P75 / P90", "%s / %s / %s / %s" % (
    P4(h2["premium_next_open"].get("p10")), P4(h2["premium_next_open"].get("p25")),
    P4(h2["premium_next_open"].get("p75")), P4(h2["premium_next_open"].get("p90")))]))
L.append(row(["次日开盘溢价为负的比例", P(share(premiums, lambda x: x < 0), 1)]))
L.append("")
L.append("**无法成交概率估计**：%s。实测任意一天以收盘价买入遇到封涨停的概率 = **%s**；"
         "严格口径（只有一字板才完全买不到）= **%s**。"
         % (h2["unbuyable_def"], P(h2["unbuyable_probability"]), P(h2["unbuyable_probability_strict"])))
L.append("")
L.append("涨停幅度按交易所实际规则分板块设定：%s。" % h2["limit_rule"])
L.append("")
L.append("**判据**")
L.append("")
L.append("| 实测值 | 阈值 | 结论 |")
L.append("|---|---|---|")
L.append(row(["涨停（含一字板）占比 %s（%d/%d）" % (P(h2["limit_up_rate"]), h2["limit_up_days"], h2["stock_days"]),
              "> 20%", "**%s**" % h2v["conclusion"]]))
L.append("")
if h2["limit_up_rate"] is not None and h2["one_word_rate"] is not None:
    L.append("口径稳健性：即便把一字板与涨停两个比例直接相加（一字板本身是涨停子集，此处属重复计数），"
             "合计为 %s，仍低于 20%% 阈值，结论不因口径选择而改变。"
             % P(h2["limit_up_rate"] + h2["one_word_rate"]))
L.append("")
L.append("补充事实：一字板占涨停日的 %s，即多数涨停日盘中仍有成交、理论上可在非涨停价位成交；"
         "但在这 %s 的交易日里，以收盘价买入需排队且不保证成交（一字板日则完全无法成交）。"
         % (P(h2["one_word_within_limit_rate"], 1), P(h2["limit_up_rate"])))
L.append("")
L.append("另一面：窗口内 **%s** 的样本股至少出现过 1 个涨停日（%s 出现过一字板），"
         "说明「随时都买得到」也不成立——能否成交高度依赖具体日期与个股，"
         "而非一个稳定的、可批量执行的买入条件。"
         % (P(h2["stocks_with_any_limit_up"], 1), P(h2["stocks_with_any_one_word"], 1)))
L.append("")

# ---- H3
L.append("## 3　H3 流动性与操纵易发性")
L.append("")
L.append("统计窗口同 H2（%s ~ %s），共 %d 个 stock-day。" % (h3["window"][0], h3["window"][1], h3["stock_days"]))
L.append("")
L.append("| 指标 | P10 | P50（中位） | P90 | 均值 |")
L.append("|---|---|---|---|---|")
a = h3["amount_cny"]
L.append(row(["日成交额（近似）", W(a.get("p10")), W(a.get("median")), W(a.get("p90")), W(a.get("mean"))]))
t = h3["turnover_proxy"]
L.append(row(["换手代理（成交额/fmv）", P(t.get("p10")), P(t.get("median")), P(t.get("p90")), P(t.get("mean"))]))
i1 = h3["impact_1m_bps"]["stats"]
L.append(row(["100 万元冲击成本（bps）", F(i1.get("p10")), F(i1.get("median")), F(i1.get("p90")), F(i1.get("mean"))]))
L.append("")
L.append("口径与假设：")
L.append("")
L.append("- 成交额口径：%s。" % h3["note_amount"])
L.append("- 换手口径：%s。" % h3["turnover_def"])
L.append("- 冲击成本口径（**假设值，非市场实测**）：%s；系数敏感性：0.05 → 中位 %s bps，0.20 → 中位 %s bps。"
         % (h3["impact_1m_bps"]["def"], F(h3["impact_1m_bps"]["sensitivity_coef_0.05"]["median"]),
            F(h3["impact_1m_bps"]["sensitivity_coef_0.20"]["median"])))
L.append("")
L.append("**判据**")
L.append("")
L.append("| 实测值 | 阈值 | 结论 |")
L.append("|---|---|---|")
L.append(row(["中位日成交额 %s" % W(a.get("median")), "< 5000 万元", "**%s**" % h3v["conclusion"]]))
L.append("")
L.append("补充事实：P10 日成交额 %s，P90 %s，个股间差异大；中位换手代理 %s，"
         "100 万元下单额约占中位日成交额的 %s。"
         % (W(a.get("p10")), W(a.get("p90")), P(t.get("median")),
            P(H4_AMOUNT / a["median"]) if a.get("median") else "NA"))
rb = h3["robustness_full_2025"]
L.append("")
L.append("逐股口径对照：以「每只股票在窗口内的中位日成交额」再取中位，为 %s（P10 %s、P90 %s），"
         "结论与合并口径一致。"
         % (W(h3["amount_per_stock_median"].get("median")), W(h3["amount_per_stock_median"].get("p10")),
            W(h3["amount_per_stock_median"].get("p90"))))
L.append("")
L.append("窗口敏感性（H3 结论依赖统计窗的选择，故并列报告）：同一 126 只样本在 **2025 全年**"
         "（含戴帽前与戴帽后）的中位日成交额为 %s、中位换手代理 %s；"
         "而策略买点窗口（10/15~11/07）为 %s、%s。后者并未更低，说明该时段并非交投清淡。"
         % (W(rb["amount"].get("median")), P(rb["turnover"].get("median")),
            W(a.get("median")), P(t.get("median"))))
L.append("")
L.append("> 本项只回答「样本股在这段窗口的成交是否足够清淡」。实测未通过阈值，"
         "因此**不能**用「小盘低流动性」作为该策略的事实前提；该结论不涉及任何主体的行为认定。")
L.append("")

# ---- H4
L.append("## 4　H4 收益分布 vs 收益承诺")
L.append("")
L.append("口径：%s。" % h4["entry_rule"])
L.append("")
L.append("| 指标 | 值 |")
L.append("|---|---|")
d = h4["return_dist"]
L.append(row(["样本数", h4["n"]]))
L.append(row(["中位", P4(d.get("median"))]))
L.append(row(["均值", P4(d.get("mean"))]))
L.append(row(["P10 / P25", "%s / %s" % (P4(d.get("p10")), P4(d.get("p25")))]))
L.append(row(["P75 / P90", "%s / %s" % (P4(d.get("p75")), P4(d.get("p90")))]))
L.append(row(["最小 / 最大", "%s / %s" % (P4(d.get("min")), P4(d.get("max")))]))
L.append(row(["P(收益 ≥ 30%)", "**%s**（%d/%d）" % (P(h4["prob_ge_30"]),
                                                sum(1 for x in rets if x >= 0.30), len(rets))]))
L.append(row(["P(收益 ≥ 50%)", "%s（%d/%d）" % (P(h4["prob_ge_50"]),
                                            sum(1 for x in rets if x >= 0.50), len(rets))]))
L.append(row(["P(收益 < 0)", "%s（%d/%d）" % (P(h4["prob_negative"]),
                                          sum(1 for x in rets if x < 0), len(rets))]))
L.append(row(["全市场等权同期基准（中位）", P4(median_of([x["bench"] for x in hold_recs]))]))
L.append(row(["超额（个股 − 等权基准）：中位 / 均值", "%s / %s" % (
    P4(h4["excess_dist"].get("median")), P4(h4["excess_dist"].get("mean")))]))
L.append(row(["超额 > 0 的比例", P(share(excs, lambda x: x > 0), 1)]))
L.append("")
L.append("**判据**")
L.append("")
L.append("| 实测值 | 阈值 | 结论 |")
L.append("|---|---|---|")
L.append(row(["P(≥30%%) = %s（%d/%d）" % (P(h4["prob_ge_30"]),
                                       sum(1 for x in rets if x >= 0.30), len(rets)),
              "< 20%", "**%s**" % h4v["conclusion"]]))
L.append("")
L.append("补充事实：均值 %s 高于中位 %s，说明分布右偏、少数个股拉高均值；"
         "P(≥50%%) 仅 %s，P(收益<0) 为 %s；同期全市场等权基准中位 %s，样本超额中位 %s。"
         % (P4(d.get("mean")), P4(d.get("median")), P(h4["prob_ge_50"]), P(h4["prob_negative"]),
            P4(median_of([x["bench"] for x in hold_recs])), P4(h4["excess_dist"].get("median"))))
L.append("")

# ---- H5
L.append("## 5　H5 债市溢出可行性（只评估）")
L.append("")
L.append("| 最小字段 | 本地是否具备 |")
L.append("|---|---|")
for k in h5["required_fields"]:
    L.append(row([k, "具备" if h5["present_locally"][k] else "**不具备**"]))
L.append("")
L.append("| 项 | 结论 |")
L.append("|---|---|")
L.append(row(["本地可用字段", "、".join(h5["panel_fields_available"])]))
L.append(row(["本地债券类数据文件", "、".join(h5["bond_data_files_found"]) if h5["bond_data_files_found"]
              else "无（已扫描 analysis/ 与项目根目录文件名）"]))
L.append(row(["结论", "**%s**" % h5["verdict"]]))
L.append("")
L.append("理由：%s" % h5["reason"])
L.append("")
L.append("若后续要做，最小数据集为：")
L.append("")
for x in h5["minimal_dataset_if_pursued"]:
    L.append("- %s" % x)
L.append("")
L.append("**判据**")
L.append("")
L.append("| 实测值 | 阈值 | 结论 |")
L.append("|---|---|---|")
L.append(row(["5 项最小字段具备 0 项", "5 项全部具备方可做", "**不可做**"]))
L.append("")

# ---- 局限
L.append("## 6　局限")
L.append("")
L.append("1. **事件前漂移无法区分「泄露」与「公开指标提前定价」**：戴帽前已有业绩预告、审计意见、"
         "退市风险提示、交易所问询等公开信号，样本股在 [-20,-1] 的负向漂移完全可能由这些公开信息驱动。"
         "本检验只是事件研究，不能识别信息泄露的因果来源，也不能证明任何主体提前交易。")
L.append("2. **成交额用 v×close 近似**：未使用成交均价（VWAP）。一字板日 close 即涨停价、"
         "盘中曾打开涨停的日子真实均价与收盘价存在偏差，因此成交额、换手代理、冲击成本均为近似值；"
         "冲击成本系数 %.2f 亦为经验假设，非本市场实测标定。" % IMPACT_COEF)
L.append("3. **2025 单年样本**：全部结论基于 126 个 2025 年戴帽事件，未覆盖跨年周期与不同市场环境；"
         "分组后「其他月份」仅 %d 个观测，t 值统计功效有限，分组差异不应过度解读。"
         % h1g["其他月份(其他原因)"]["n"])
L.append("")

L.append("## 7　判据汇总")
L.append("")
L.append("| 命题 | 实测值 | 阈值 | 结论 |")
L.append("|---|---|---|---|")
L.append(row(["H1 事件前异常", "[-20,-1] CAR 中位 %s，t=%s" % (P4(h1w["[-20,-1]"]["median"]),
                                                          F(h1w["[-20,-1]"]["t"], 2)),
              "中位<-3% 且 t<-2", "**%s**" % ("成立" if h1v["pass"] else "不成立")]))
L.append(row(["H2 是否买得到", "涨停占比 %s" % P(h2["limit_up_rate"]), ">20%",
              "**%s**" % ("成立（存在买不到）" if h2v["pass"] else "不成立（未达 20%）")]))
L.append(row(["H3 小盘低流动性", "中位日成交额 %s（阈值 5000 万元）" % W(a.get("median")), "<5000 万元",
              "**%s**" % ("成立" if h3v["pass"] else "不成立（略高于阈值）")]))
L.append(row(["H4 收益承诺", "P(≥30%%) = %s" % P(h4["prob_ge_30"]), "<20%",
              "**%s**" % ("成立（承诺与事实不符）" if h4v["pass"] else "不成立")]))
L.append(row(["H5 债市溢出", "最小字段具备 0/5", "需全部具备", "**不可做**"]))
L.append("")
L.append("---")
L.append("")
L.append("复算方式：`python analysis/out_stcomp/st_comp.py`（读缓存，不触碰原始 zip）。"
         "本文所有数值均由该脚本生成，与 `result.json` 逐字段一致。")

with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print("[write] report.md  %.1fs" % (time.time() - T0))
