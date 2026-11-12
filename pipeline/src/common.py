# -*- coding: utf-8 -*-
"""通用工具：截面统计、IC、Newey-West、分层、成本、IO。
只依赖 numpy。所有函数都不含未来信息（调用方负责窗口）。
"""
import json
import os
import pickle

import numpy as np


# ----------------------------- IO -----------------------------

def save_pkl(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=4)


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------- 截面处理 -----------------------------

def _rank(x):
    """平均秩（处理并列），返回 1..n 的 float 数组。"""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    # 并列取平均秩
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def winsorize(s, lo=1.0, hi=99.0):
    """分位截尾，返回新数组。"""
    s = np.asarray(s, dtype=float)
    if len(s) < 5:
        return s.copy()
    a, b = np.percentile(s, [lo, hi])
    return np.clip(s, a, b)


def cs_rank(s):
    """截面百分位 0-1（越大越高）。"""
    s = np.asarray(s, dtype=float)
    if len(s) == 0:
        return s
    return (_rank(s) - 1) / max(len(s) - 1, 1)


def cs_zscore(s):
    s = np.asarray(s, dtype=float)
    sd = np.std(s, ddof=0)
    if sd < 1e-12:
        return np.zeros_like(s)
    return (s - np.mean(s)) / sd


def neutralize(values, groups, sizes=None):
    """
    对 values 做「行业哑变量 + ln(市值)」截面回归取残差。
    groups: 与 values 等长的行业标签列表
    sizes:  与 values 等长的市值（可选）
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 10:
        return v.copy()
    g = np.asarray([str(x) for x in groups])
    uniq = list(dict.fromkeys(g.tolist()))
    cols = [np.ones(n)]
    for u in uniq[1:]:                       # 去掉一组避免共线
        cols.append((g == u).astype(float))
    if sizes is not None:
        sz = np.log(np.maximum(np.asarray(sizes, dtype=float), 1.0))
        if np.std(sz) > 1e-12:
            cols.append(cs_zscore(sz))
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    return v - X @ beta


# ----------------------------- 统计 -----------------------------

def pearson(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return np.nan
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return np.nan
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (sa * sb))


def spearman(a, b):
    """秩相关。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3:
        return np.nan
    return pearson(_rank(a), _rank(b))


def nw_t(x, lag):
    """Newey-West t 值（均值是否为 0）。lag 为重叠期数。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 5:
        return np.nan
    mu = x.mean()
    e = x - mu
    s0 = np.dot(e, e) / n
    for L in range(1, min(lag, n - 1) + 1):
        w = 1.0 - L / (lag + 1.0)
        cov = np.dot(e[L:], e[:-L]) / n
        s0 += 2 * w * cov
    se = np.sqrt(max(s0, 1e-18) / n)
    return float(mu / se)


def max_drawdown(nav):
    nav = np.asarray(nav, dtype=float)
    if len(nav) == 0:
        return np.nan
    peak = np.maximum.accumulate(nav)
    return float((nav / peak - 1).min())


def annualize(period_ret, periods_per_year):
    return float((1 + np.mean(period_ret)) ** periods_per_year - 1)


def sharpe(period_ret, periods_per_year):
    r = np.asarray(period_ret, dtype=float)
    sd = np.std(r, ddof=1)
    if sd < 1e-12:
        return np.nan
    return float(np.mean(r) / sd * np.sqrt(periods_per_year))


def quantile_bins(values, k):
    """按分位切 k 组，返回组号数组（0 最小）。并列不影响。"""
    v = np.asarray(values, dtype=float)
    r = _rank(v) / len(v)
    b = np.floor(r * k).astype(int)
    return np.clip(b, 0, k - 1)
