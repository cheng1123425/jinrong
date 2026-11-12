#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60 分钟「分型自相似」参考实现（因子建模师 × 小波顾问 联合规范的可执行版）

实现《算法规范_分型自相似.md》：
  A. 缠论分型序列匹配（包含关系处理 → 分型 → 幅度/间隔余弦）
  B. MODWT 多尺度形状相似（Haar，1/2 归一化）
  C. 双确认融合

用法：
  python pattern_similarity.py --selftest
  python pattern_similarity.py --csv "tmp60/60分钟/sz300159.csv"
  python pattern_similarity.py --dir "tmp60/60分钟" --limit 30

注意：这是规范的可执行版，不是可直接上线的策略。参数写死在 PARAMS，禁止按结果回调。
"""

import argparse
import csv
import os
import sys

import numpy as np

PARAMS = {
    "K": 5,              # 匹配分型个数
    "L": 480,            # 历史搜索长度（60分钟根数，约120交易日）
    "amp_theta": 0.8,    # 幅度余弦阈值
    "gap_theta": 0.7,    # 间隔余弦阈值
    "theta": 0.8,        # 方案A综合阈值
    "W": 64,             # 小波窗口
    "J": 3,              # MODWT 层数
    "wave_theta": 0.75,  # 方案B阈值
}


# ---------------------------------------------------------------- A. 分型

def merge_inclusion(bars):
    """bars: [(date,time,o,h,l,c)] → (无包含序列, 原始索引映射)。

    合并会改变长度，必须记录每根合并 K 线对应原始 bars 的最后一根索引，
    否则后续按分型位置取原始收盘价会错位。
    """
    if not bars:
        return [], []
    out = [list(bars[0])]
    mapping = [0]
    trend = 1
    for idx, b in enumerate(bars[1:], start=1):
        p = out[-1]
        inc = (b[3] >= p[3] and b[4] <= p[4]) or (b[3] <= p[3] and b[4] >= p[4])
        if inc:
            if trend >= 0:
                h, l = max(b[3], p[3]), max(b[4], p[4])
            else:
                h, l = min(b[3], p[3]), min(b[4], p[4])
            out[-1] = [p[0], p[1], p[2], h, l, b[5]]
            mapping[-1] = idx
        else:
            trend = 1 if b[3] > p[3] else (-1 if b[3] < p[3] else trend)
            out.append(list(b))
            mapping.append(idx)
    return out, mapping


def find_fractals(B, mapping):
    """返回 [(pos_in_B, type, price, t_original)]，type: 顶=+1 底=-1，保证顶底交替。"""
    raw = []
    for i in range(1, len(B) - 1):
        h0, l0 = B[i - 1][3], B[i - 1][4]
        h1, l1 = B[i][3], B[i][4]
        h2, l2 = B[i + 1][3], B[i + 1][4]
        if h1 > h0 and h1 > h2 and l1 > l0 and l1 > l2:
            raw.append([i, 1, h1, mapping[i]])
        elif l1 < l0 and l1 < l2 and h1 < h0 and h1 < h2:
            raw.append([i, -1, l1, mapping[i]])
    out = []
    for f in raw:
        if out and out[-1][1] == f[1]:
            if f[1] == 1 and f[2] >= out[-1][2]:
                out[-1] = f
            elif f[1] == -1 and f[2] <= out[-1][2]:
                out[-1] = f
        else:
            out.append(f)
    return out


def _vectors(seq):
    amp, gap = [], []
    for a, b in zip(seq, seq[1:]):
        amp.append(abs(b[2] - a[2]) / max(abs(a[2]), 1e-9))
        gap.append(max(b[3] - a[3], 1))
    return np.asarray(amp, float), np.asarray(gap, float)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def fractal_similarity(now, hist, p=PARAMS):
    if [x[1] for x in now] != [x[1] for x in hist]:
        return None
    a1, g1 = _vectors(now)
    a2, g2 = _vectors(hist)
    ac, gc = _cos(a1, a2), _cos(g1, g2)
    if ac < p["amp_theta"] or gc < p["gap_theta"]:
        return None
    return 0.5 * ac + 0.5 * gc, ac, gc


# ---------------------------------------------------------------- B. 小波

def modwt_haar(x, J):
    """Haar MODWT，归一化 1/2，能量守恒。"""
    a = np.asarray(x, float).copy()
    W = []
    for j in range(1, J + 1):
        s = 2 ** (j - 1)
        W.append(0.5 * (a - np.roll(a, s)))
        a = 0.5 * (a + np.roll(a, s))
    return W, a


def wave_features(closes, p=PARAMS):
    W, A = modwt_haar(np.asarray(closes, float), p["J"])
    return W + [A]


def wave_similarity(f1, f2, p=PARAMS):
    weights = [0.15, 0.25, 0.30, 0.30][:len(f1)]
    total, used = 0.0, 0.0
    for w, a, b in zip(weights, f1, f2):
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            continue
        c = _cos(a - a.mean(), b - b.mean())
        total += w * c
        used += w
    if used < 1e-9:
        return 0.0
    return total / used


# ---------------------------------------------------------------- C. 信号

def build_signal(bars, idx, closes, p=PARAMS):
    """在 idx（当前 60 分钟根）判断是否存在分型自相似。只用 bars[:idx+1]。"""
    if idx < p["W"] + 10:
        return None
    B, mp = merge_inclusion(bars[:idx + 1])
    fr = find_fractals(B, mp)
    K = p["K"]
    if len(fr) < K + 2:
        return None
    now = fr[-K:]
    # 当前分型必须已由其后一根确认
    if now[-1][0] + 1 > len(B) - 1:
        return None
    best = None
    for end in range(len(fr) - K - 1, K - 1, -1):
        hist = fr[end - K:end]
        if len(hist) < K:
            continue
        if now[0][3] - hist[-1][3] > p["L"]:
            continue
        r = fractal_similarity(now, hist, p)
        if r and (best is None or r[0] > best[0]):
            best = (r[0], hist)
    if best is None or best[0] < p["theta"]:
        return None
    # 小波确认：历史窗口末尾对齐到该分型对应的原始 K 线
    w_now = wave_features(closes[idx - p["W"] + 1: idx + 1], p)
    h_abs = best[1][-1][3]
    if h_abs + 1 < p["W"]:
        return None
    w_hist = wave_features(closes[h_abs - p["W"] + 1: h_abs + 1], p)
    ws = wave_similarity(w_now, w_hist, p)
    if ws < p["wave_theta"]:
        return None
    return {"score": 0.5 * best[0] + 0.5 * ws,
            "fractal_sim": best[0], "wave_sim": ws,
            "hist_end": best[1][-1][0], "date": bars[idx][0], "time": bars[idx][1],
            "close": closes[idx]}


# ---------------------------------------------------------------- IO

def read60(path):
    bars, closes = [], []
    with open(path, encoding="utf-8-sig") as f:
        rd = csv.reader(f)
        next(rd, None)
        for r in rd:
            if len(r) < 7:
                continue
            try:
                o, h, l, c = float(r[2]), float(r[3]), float(r[4]), float(r[5])
            except ValueError:
                continue
            bars.append((r[0], r[1], o, h, l, c))
            closes.append(c)
    return bars, np.asarray(closes, float)


def scan(bars, closes, p=PARAMS, step=4):
    out = []
    for i in range(p["W"] + 10, len(bars), step):
        s = build_signal(bars, i, closes, p)
        if s:
            out.append(s)
    # 同日去重，保留分数最高
    seen = {}
    for s in out:
        k = s["date"]
        if k not in seen or s["score"] > seen[k]["score"]:
            seen[k] = s
    return sorted(seen.values(), key=lambda x: x["date"])


# ---------------------------------------------------------------- 自检

def selftest(verbose=True):
    ok = True

    def chk(name, cond, info=""):
        nonlocal ok
        ok &= bool(cond)
        if verbose:
            print(f"[{'OK ' if cond else 'FAIL'}] {name} {info}")

    # 1) 包含处理与分型
    bars = []
    prices = [10, 11, 12, 11, 10, 9, 10, 11, 12, 13, 12, 11, 10, 9, 8, 9, 10, 11]
    for i, c in enumerate(prices):
        bars.append((f"2025-01-{i//4+1:02d}", f"{i%4}", c, c + 0.2, c - 0.2, c))
    B, mp = merge_inclusion(bars)
    fr = find_fractals(B, mp)
    chk("分型识别非空", len(fr) >= 3, f"{len(fr)} 个")
    chk("顶底交替", all(fr[i][1] != fr[i + 1][1] for i in range(len(fr) - 1)))

    # 2) 相同序列相似度=1
    now = [[0, 1, 12, 0], [1, -1, 10, 4], [2, 1, 13, 8], [3, -1, 11, 12], [4, 1, 14, 16]]
    r = fractal_similarity(now, [x[:] for x in now])
    chk("同序列相似度≈1", r is not None and abs(r[0] - 1) < 1e-9, f"{r[0]:.4f}")

    # 3) 类型不一致 → None
    bad = [x[:] for x in now]; bad[1][1] = 1
    chk("类型不一致返回None", fractal_similarity(now, bad) is None)

    # 4) 幅度结构不同 → 应被阈值拒绝（amp 余弦 < 0.8）
    alt = [[0, 1, 12, 0], [1, -1, 18.0, 4], [2, 1, 16.2, 8], [3, -1, 22.7, 12], [4, 1, 21.6, 16]]
    r2 = fractal_similarity(now, alt)
    chk("幅度结构不同被拒绝", r2 is None, f"{r2[0]:.3f}" if r2 else "None")

    # 5) MODWT 能量守恒
    rng = np.random.default_rng(7)
    x = rng.normal(size=64)
    W, A = modwt_haar(x, 3)
    e = sum(np.sum(w ** 2) for w in W) + np.sum(A ** 2)
    chk("MODWT 能量守恒", abs(e - np.sum(x ** 2)) < 1e-8, f"{abs(e-np.sum(x**2)):.2e}")

    # 6) 相同窗口小波相似度≈1
    ws = wave_similarity(wave_features(x), wave_features(x))
    chk("小波自相似≈1", abs(ws - 1) < 1e-9, f"{ws:.4f}")

    # 7) 随机序列不应大量触发
    out = 0
    for _ in range(20):
        y = np.cumsum(rng.normal(0, 0.01, 500)) + 10
        bb = [(f"d{i//4}", f"{i%4}", y[i], y[i] + .01, y[i] - .01, y[i]) for i in range(len(y))]
        out += len(scan(bb, y, step=8))
    checks = 20 * len(range(PARAMS["W"] + 10, 500, 8))
    rate = out / max(checks, 1)
    chk("随机数据信号率 < 5%", rate < 0.05, f"{out}/{checks} = {rate:.1%}（假阳性率）")

    if verbose:
        print("\n自检结果:", "全部通过 ✅" if ok else "存在失败 ❌")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--dir")
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    if args.csv:
        bars, closes = read60(args.csv)
        sigs = scan(bars, closes)
        print(f"{os.path.basename(args.csv)} | {len(bars)} 根 | 信号 {len(sigs)} 个")
        for s in sigs:
            print(f"  {s['date']} {s['time']} 收{s['close']:.2f} 分数{s['score']:.3f} "
                  f"(分型{s['fractal_sim']:.2f}/小波{s['wave_sim']:.2f})")
        return

    if args.dir:
        files = [f for f in os.listdir(args.dir) if f.endswith(".csv")][:args.limit]
        total = 0
        for f in files:
            bars, closes = read60(os.path.join(args.dir, f))
            n = len(scan(bars, closes))
            total += n
            if n:
                print(f"  {f}: {n} 个信号")
        print(f"目录 {args.dir} 前 {len(files)} 只 | 信号合计 {total}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
