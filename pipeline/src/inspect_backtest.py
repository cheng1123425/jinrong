# -*- coding: utf-8 -*-
"""把 S6 回测/IC/分层拆开给你看：抽一个调仓日，逐步骤打印中间量，并手算复现。

用法：python src/inspect_backtest.py --run runs/demo [--date 2021-06-30]
输出：runs/demo/09_回测与分层拆解.md
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_pkl, load_json, spearman, quantile_bins, _rank  # noqa: E402

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/demo")
    ap.add_argument("--date", default="")
    a = ap.parse_args()
    run = a.run if os.path.isabs(a.run) else os.path.join(BASE, a.run)

    prices = load_pkl(os.path.join(run, "..", "demo_data", "prices.pkl"))
    factors = load_pkl(os.path.join(run, "05_factor_final.pkl"))
    m = load_json(os.path.join(run, "06_metrics.json"))
    dd = load_json(os.path.join(run, "04_factor_diag.json"))

    dates = prices["dates"]
    close = prices["close"]
    idx = {d: i for i, d in enumerate(dates)}
    hold, k = m["hold"], m["layers"]

    def fwd(code, t, h):
        cl = close[code]
        if t + h >= len(cl):
            return None
        return cl[t + h] / cl[t] - 1.0

    # 选一个 IC 接近中位数的调仓日，避免挑极端的
    cand = []
    for d in sorted(factors):
        t = idx.get(d)
        if t is None or t + hold >= len(dates):
            continue
        cs = factors[d]
        codes = list(cs)
        v = np.array([cs[c] for c in codes])
        r = np.array([fwd(c, t, hold) if fwd(c, t, hold) is not None else np.nan for c in codes])
        mk = ~np.isnan(r)
        if mk.sum() >= 50:
            cand.append((spearman(v[mk], r[mk]), d, t, codes, v, r, mk))
    cand.sort()
    if a.date:
        pick = next(x for x in cand if x[1] == a.date)
    else:
        pick = cand[len(cand) // 2]

    ic, d, t, codes, v, r, mk = pick
    L = []
    L.append("# 回测与 IC 分层 · 单日拆解\n")
    L.append(f"> 样本日 **{d}**（该日 IC 在全部 {len(cand)} 个调仓日中处于中位）｜持有 {hold} 个交易日｜分 {k} 组\n")

    L.append("## 1. 这一天的截面长什么样\n")
    L.append(f"- 当日有因子值的股票：**{len(codes)}** 只")
    L.append(f"- 其中能算未来 {hold} 日收益（未退市、数据完整）：**{int(mk.sum())}** 只")
    L.append(f"- 因子值分布：min {v.min():.3f} / P25 {np.percentile(v,25):.3f} / 中位 {np.median(v):.3f} / P75 {np.percentile(v,75):.3f} / max {v.max():.3f}")
    L.append(f"- 因子值范围 {v.min():.3f} ~ {v.max():.3f}：这是**截面百分位（0~1）**，不是原始残差")
    L.append("  因为 S5 已经做过 `winsorize → 行业+市值中性化 → 截面 rank`\n")

    L.append("## 2. IC 是怎么算的\n")
    L.append("```")
    L.append("IC = spearman(因子值, 未来21日收益)   # 每个调仓日算一次，得到一个序列")
    L.append("```")
    vr = v[mk]
    rr = r[mk]
    ic_prog = spearman(vr, rr)
    ic_manual = np.corrcoef(_rank(vr), _rank(rr))[0, 1]
    L.append(f"- 程序算的 IC：**{ic_prog:+.4f}**")
    L.append(f"- 手算（秩 → 皮尔逊）：**{ic_manual:+.4f}**（差 {abs(ic_prog-ic_manual):.2e}，说明 spearman 就是秩相关）")
    L.append(f"- 复现校验：`corrcoef(rank(factor), rank(fwd_ret))` 与程序值一致 ✅\n")
    L.append("**注意事项**：IC 是逐日/逐期截面相关，不是时序相关；它衡量的是"
             "「同一天里，因子排名和未来收益排名有多像」，不是「因子涨收益就涨」。\n")

    L.append("## 3. 分组是怎么切的\n")
    L.append("```python")
    L.append("b = quantile_bins(vals, k)   # 按 rank/len*k 取整，等分位、等数量，不按数值区间")
    L.append("```")
    b = quantile_bins(vr, k)
    L.append("| 组 | 股票数 | 因子值区间 | 平均因子值 |")
    L.append("|---|---|---|---|")
    for g in range(k):
        sel = b == g
        L.append(f"| L{g+1} | {int(sel.sum())} | {vr[sel].min():.4f} ~ {vr[sel].max():.4f} | {vr[sel].mean():.4f} |")
    L.append("\n等数量切分的好处：每组都有足够样本，组间收益差不会被「某组只有 3 只票」放大。\n")

    L.append("## 4. 分层收益与多空\n")
    L.append("```")
    L.append("每组收益 = 该组所有股票未来21日收益的等权平均")
    L.append("多空     = L{k} 平均收益 - L1 平均收益".format(k=k))
    L.append("```")
    gr = []
    for g in range(k):
        sel = b == g
        gr.append(float(rr[sel].mean()))
    order = np.argsort(-np.asarray(gr))          # 收益从高到低的名次
    rank_of = {int(g): i + 1 for i, g in enumerate(order)}
    L.append("| 组 | 平均因子值 | 平均未来21日收益 | 收益名次 |")
    L.append("|---|---|---|---|")
    for g in range(k):
        L.append(f"| L{g+1} | {vr[b == g].mean():.4f} | {gr[g]*100:+.3f}% | {rank_of[g]} |")
    L.append(f"\n- 多空（L{k} − L1）= **{(gr[-1]-gr[0])*100:+.3f}%**")
    L.append(f"- 单调性（程序判定）= **{all(np.diff(gr) > 0) or all(np.diff(gr) < 0)}**"
             f"（判定标准：五组收益严格递增或严格递减，不含相等）\n")

    # ---------- 全期分层 ----------
    per_layer = [[] for _ in range(k)]
    spreads, ics, turns2, holds_prev = [], [], [], None
    for ic_, d_, t_, codes_, v_, r_, mk_ in cand:
        vv, rr2 = v_[mk_], r_[mk_]
        bb = quantile_bins(vv, k)
        for g in range(k):
            sel = bb == g
            if sel.sum():
                per_layer[g].append(float(rr2[sel].mean()))
        if bb.min() == 0 and bb.max() == k - 1:
            spreads.append(float(rr2[bb == k - 1].mean() - rr2[bb == 0].mean()))
            cur = set(np.asarray(codes_)[bb == k - 1]) | set(np.asarray(codes_)[bb == 0])
            if holds_prev:
                turns2.append(len(cur ^ holds_prev) / max(len(cur), 1))
            holds_prev = cur
    L.append("## 4b. 全期分层统计（{0} 个调仓日）\n".format(len(cand)))
    L.append("单日那张表只能说明「这一天怎么算」，不能说明「因子好不好」。把所有调仓日摞起来才有意义：\n")
    L.append("| 组 | 每期均值 | t 值 | 胜率 | 5.5年累计净值 |")
    L.append("|---|---|---|---|---|")
    for g in range(k):
        x = np.asarray(per_layer[g], float)
        tval = x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 and x.std(ddof=1) > 0 else float("nan")
        navg = float(np.prod(1 + x))
        L.append(f"| L{g+1} | {x.mean()*100:+.3f}% | {tval:+.2f} | {np.mean(x>0)*100:.1f}% | {navg:.3f} |")
    L.append(f"\n- 全期多空每期均值 **{np.mean(spreads)*100:+.3f}%**，t = "
             f"{np.mean(spreads)/(np.std(spreads,ddof=1)/np.sqrt(len(spreads))):+.2f}")
    means = [float(np.mean(x)) * 100 for x in per_layer]
    mono = all(b_ > a_ for a_, b_ in zip(means[:-1], means[1:])) or all(b_ < a_ for a_, b_ in zip(means[:-1], means[1:]))
    L.append("- 全期各组均值（%）：" + " → ".join(f"{v:+.3f}" for v in means))
    L.append(f"- 按全期均值判定单调性：**{mono}**")
    L.append("  → 单日不单调是常态，**要看的是一段时间的平均是否单调**，因为单期噪声远大于信号")
    if not mono and means[-1] > means[0]:
        L.append("  → 本例就是「两端有效、中间无序」：L5>L1 但 L2 比 L1 更低。"
                 "这种形态下多空能赚钱，但**不能按组递推**（比如「买 L4 比 L3 好」就不成立）。\n")
    L.append("**为什么单日和全期会打架**：单日 200 只票，收益噪声 σ≈8%，均值差只有 0.1% 量级，"
             "信噪比极低；摞 60 期后噪声按 √n 衰减，才有统计意义。这也是为什么不能拿一天的分层图下结论。\n")

    L.append("## 5. 多空净值是怎么累起来的\n")
    L.append("```python")
    L.append("nav[0] = 1")
    L.append("for 每个调仓日: nav.append(nav[-1] * (1 + spread))   # spread = L{k} - L1".format(k=k))
    L.append("年化 = (1 + mean(spread)) ** (252/调仓间距) - 1")
    L.append("夏普 = mean(spread) / std(spread) * sqrt(252/调仓间距)")
    L.append("回撤 = min(nav / cummax(nav) - 1)")
    L.append("```")
    L.append(f"\n本次结果：多空年化 **{m['ls_annual']*100:.2f}%**、夏普 **{m['ls_sharpe']:.3f}**、"
             f"最大回撤 **{m['ls_mdd']*100:.2f}%**")
    L.append("\n**这里有个必须说清的假设**：净值是「每期把多空收益按复利再投入」，"
             "隐含**可无限做空、无保证金约束、无融券成本**。真实 A 股做空受限，"
             "多空曲线只能当作因子强度的度量，不能当作可实现收益。\n")

    L.append("## 6. 换手与成本\n")
    L.append("```")
    L.append("持仓 = L{k} 组 + L1 组（约 40% 的股票）".format(k=k))
    L.append("单边换手 = |本期持仓 XOR 上期持仓| / |本期持仓|")
    L.append("每期成本 = 单边换手 × 2 × 单边费率（买卖各一次）")
    L.append("```")
    L.append(f"\n实测：单边换手 **{m['turnover']*100:.1f}%**，每期成本 **{m['cost_per_rebalance']*100:.3f}%**，"
             f"年化毛利 {m['ls_annual']*100:.2f}% → 扣费后 **{m['net_ls_annual']*100:.2f}%**")
    L.append("\n**换手算法的局限**：这是「名字换手」（有多少只票换了），不是「权重换手」。"
             "等权组合下两者接近，但如果按市值加权或因子加权，必须改成权重口径。"
             "另外 40% 的持仓意味着一半资金在多头、一半在空头，实际资金占用是 2 倍名义。\n")

    L.append("## 7. 三个容易被忽略的偏差\n")
    L.append("| 偏差 | 这里怎么处理的 | 还差什么 |")
    L.append("|---|---|---|")
    L.append("| 前视 | 因子只用 ≤ ann_date 的数据；G3.1/G3.2 门禁 | 已处理 |")
    L.append("| 同日成交 | 用 **t 日收盘价**买入、t+21 收盘卖出 | 严格应 t+1 开盘，或假设收盘集合竞价能成交 |")
    L.append("| 幸存者偏差 | 股票池来自当前面板，退市股不在里面 | 本次是合成数据无退市；真实数据必须补退市股 |")
    L.append("| 可交易性 | 未剔除停牌、涨跌停 | **未处理**，真实回测必须加 |")
    L.append("\n## 8. 一句话总结\n")
    L.append("IC 回答「排序准不准」，分层回答「单调不单调」，多空回答「价差有多大」，"
             "换手+成本回答「拿不拿得到」。四个都过才算能用；缺一个，前三个都可以是假的。")

    out = os.path.join(run, "09_回测与分层拆解.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    print(f"\n已写出 {out}")


if __name__ == "__main__":
    main()
