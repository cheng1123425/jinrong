# -*- coding: utf-8 -*-
"""合成演示数据：植入一个已知强度的「预期外现金流」信号，用于验证流水线与门禁。

生成两个文件：
  prices.pkl  {dates, close{code:[...]}, meta{code:{industry,mktcap}}, codes}
  reports.pkl {rows:[{code, period, season, ann_date, cfo_ytd, ar, br, ap, inv}]}

注意：这是**合成数据**，任何数值都不得当作真实市场结论。
"""
import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import save_pkl  # noqa: E402

SEED = 20261113
N_CODES = 200
N_IND = 10
Q_START = (2018, 1)
N_Q = 28            # 2018Q1 ~ 2024Q4
SIGNAL = 0.030      # 每 1 个标准差意外，未来 21 日超额 3%（植入强度）


def quarter_ends(y0, q0, n):
    out = []
    y, q = y0, q0
    for _ in range(n):
        m = q * 3
        # 季末最后一天
        if m == 12:
            d = dt.date(y, 12, 31)
        else:
            d = dt.date(y, m + 1, 1) - dt.timedelta(days=1)
        out.append((y, q, d))
        q += 1
        if q > 4:
            q = 1
            y += 1
    return out


def business_days(a, b):
    out = []
    d = a
    while d <= b:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def build(out_dir):
    rng = np.random.default_rng(SEED)
    dates = business_days(dt.date(2018, 1, 1), dt.date(2025, 6, 30))
    T = len(dates)
    idx_of = {d: i for i, d in enumerate(dates)}

    codes = [f"s{i:04d}" for i in range(N_CODES)]
    inds = [f"IND{j}" for j in range(N_IND)]
    meta = {}
    for i, c in enumerate(codes):
        meta[c] = {"industry": inds[rng.integers(0, N_IND)],
                   "mktcap": float(np.exp(rng.normal(3.5, 1.0)) * 1e5),
                   # 确定性标记，避免改变 RNG 序列
                   "is_financial": (i % 13 == 0),
                   "is_st": (i % 17 == 0)}

    quarters = quarter_ends(Q_START[0], Q_START[1], N_Q)
    R = rng.normal(0, 0.014, size=(T, N_CODES))

    rows = []
    planted = {}
    for ci, c in enumerate(codes):
        sigma = float(np.exp(rng.normal(0, 0.4)))
        prev = 0.0
        prev_ytd = {}
        level_ar = 500.0
        for (y, q, pend) in quarters:
            shock = float(rng.normal(0, 1.0)) * sigma
            cfo_s = 0.45 * prev + shock
            prev = cfo_s
            # 应收账款等随经营规模缓慢漂移，与 cfo 弱相关
            level_ar = 0.85 * level_ar + 0.3 * cfo_s + float(rng.normal(0, 20))
            ar = max(level_ar + rng.normal(0, 15), 1.0)
            br = max(0.25 * ar + rng.normal(0, 8), 1.0)
            ap = max(0.18 * ar + rng.normal(0, 6), 1.0)
            inv = max(0.60 * ar + rng.normal(0, 12), 1.0)

            prev_ytd[y] = prev_ytd.get(y, 0.0) + cfo_s
            cfo_ytd = prev_ytd[y]

            # 公告日：季末后 60~110 天
            lag = int(60 + 50 * rng.random())
            ann = pend + dt.timedelta(days=lag)
            ann_s = ann.isoformat()

            z = shock / (sigma + 1e-12)
            planted[(c, ann_s)] = z

            # 在公告日之后 21 个交易日植入漂移
            if ann_s in idx_of:
                t0 = idx_of[ann_s]
                t1 = min(t0 + 21, T)
                if t1 > t0:
                    R[t0:t1, ci] += (SIGNAL * z) / (t1 - t0)

            rows.append({"code": c, "period": pend.isoformat(), "season": q,
                         "ann_date": ann_s, "cfo_ytd": round(cfo_ytd, 4),
                         "ar": round(ar, 4), "br": round(br, 4),
                         "ap": round(ap, 4), "inv": round(inv, 4)})

    # 5% 随机缺失，测试丢弃率披露
    for r in rows:
        if rng.random() < 0.05:
            r["cfo_ytd"] = None

    close = {}
    for ci, c in enumerate(codes):
        close[c] = [round(float(x), 6) for x in (100.0 * np.cumprod(1 + R[:, ci]))]

    os.makedirs(out_dir, exist_ok=True)
    save_pkl({"dates": dates, "close": close, "meta": meta, "codes": codes},
             os.path.join(out_dir, "prices.pkl"))
    save_pkl({"rows": rows, "planted_z": planted},
             os.path.join(out_dir, "reports.pkl"))
    print(f"prices: {T} 天 × {N_CODES} 只 | reports: {len(rows)} 条 | 季度 {quarters[0][0]}Q{quarters[0][1]} ~ {quarters[-1][0]}Q{quarters[-1][1]}")
    print(f"植入信号强度: {SIGNAL:.3f} / 1σ | seed={SEED}")
    return os.path.join(out_dir, "prices.pkl"), os.path.join(out_dir, "reports.pkl")


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build(os.path.join(base, "runs", "demo_data"))
