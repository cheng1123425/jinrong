# -*- coding: utf-8 -*-
import pickle, json, statistics as st, collections
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = pickle.load(open(A + '/panel2025.pkl', 'rb'))
ev = json.load(open(A + '/st_events.json', encoding='utf-8'))

def q(xs, p):
    xs = sorted(xs); 
    if not xs: return float('nan')
    k = (len(xs) - 1) * p; f = int(k); c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)

def ret_after(d, i0, T):
    if i0 + T >= len(d['c']) or i0 < 0: return None
    return d['c'][i0 + T] / d['c'][i0] - 1

# 2025 新增 ST
new_st = {k: v for k, v in ev.items() if v['st_date'] and k in panel}
print('新增ST且在面板:', len(new_st))

# A. 戴帽后收益
rows = []
for code, v in new_st.items():
    d = panel[code]
    try: i0 = d['dates'].index(v['st_date'])
    except ValueError: continue
    r = {T: ret_after(d, i0, T) for T in (5, 10, 20, 40, 60)}
    rows.append((code, v, i0, r))
print(f'\n=== A. 戴帽后收益（n={len(rows)}）===')
for T in (5, 10, 20, 40, 60):
    xs = [r[T] for _, _, _, r in rows if r[T] is not None]
    if not xs: continue
    win = sum(1 for x in xs if x > 0) / len(xs)
    print(f'  T+{T:<2d}  n={len(xs):3d}  均值 {st.mean(xs)*100:+6.2f}%  中位 {q(xs,0.5)*100:+6.2f}%  胜率 {win*100:5.1f}%  最差 {min(xs)*100:+6.1f}%')

# B. 2025 年最低点分布（限定 5/1 之后，避免年初低点）
print('\n=== B. 戴帽年份最低点（5/1 之后）===')
mon = collections.Counter(); dec = collections.Counter(); lowdates = []
for code, v, i0, r in rows:
    d = panel[code]
    idxs = [i for i, dt in enumerate(d['dates']) if dt >= '2025-05-01']
    if not idxs: continue
    j = min(idxs, key=lambda i: d['l'][i])
    dt = d['dates'][j]; lowdates.append((code, dt, d['l'][j]))
    mon[dt[:7]] += 1
    day = int(dt[8:10])
    dec[dt[:8] + ('上旬' if day <= 10 else '中旬' if day <= 20 else '下旬')] += 1
for k in sorted(mon): print(f'  {k}: {mon[k]:3d}')
print('  --- 6月内按旬 ---')
for k in sorted(k for k in dec if k.startswith('2025-06')): print(f'  {k}: {dec[k]:3d}')
in_610 = sum(1 for _, dt, _ in lowdates if '2025-06-01' <= dt <= '2025-06-20')
in_jun = sum(1 for _, dt, _ in lowdates if dt[:7] == '2025-06')
print(f'  6月见底: {in_jun}/{len(lowdates)} = {in_jun/len(lowdates)*100:.1f}%')
print(f'  6/1-6/20 见底: {in_610}/{len(lowdates)} = {in_610/len(lowdates)*100:.1f}%')

# C. 低点 -> 10/31 涨幅
print('\n=== C. 6月低点 -> 2025-10-31 涨幅 ===')
gains = []
for code, dt_low, _ in lowdates:
    d = panel[code]
    if not dt_low[:7] == '2025-06': continue
    i_low = d['dates'].index(dt_low)
    i_end = max(i for i, dt in enumerate(d['dates']) if dt <= '2025-10-31')
    if i_end <= i_low: continue
    gains.append(d['c'][i_end] / d['c'][i_low] - 1)
if gains:
    print(f'  n={len(gains)} 均值 {st.mean(gains)*100:+.1f}% 中位 {q(gains,0.5)*100:+.1f}%')
    print(f'  分位: 10% {q(gains,0.1)*100:+.1f}% | 25% {q(gains,0.25)*100:+.1f}% | 75% {q(gains,0.75)*100:+.1f}% | 90% {q(gains,0.9)*100:+.1f}%')
    print(f'  涨超30%: {sum(1 for g in gains if g>0.3)}/{len(gains)} = {sum(1 for g in gains if g>0.3)/len(gains)*100:.1f}%')
    print(f'  下跌: {sum(1 for g in gains if g<0)}/{len(gains)} = {sum(1 for g in gains if g<0)/len(gains)*100:.1f}%')

# D. 10/15-11/07 买入持有 20 日：ST vs 全市场
def window_buy(panel_subset, lo, hi, hold=20):
    out = []
    for code, d in panel_subset.items():
        cand = [i for i, dt in enumerate(d['dates']) if lo <= dt <= hi]
        for i in cand:
            if i + hold < len(d['c']):
                out.append(d['c'][i + hold] / d['c'][i] - 1)
                break
    return out

st_panel = {k: panel[k] for k in new_st if k in panel}
xs = window_buy(st_panel, '2025-10-15', '2025-11-07')
print(f'\n=== D. 10/15-11/07 任意日买入、持有20日 ===')
if xs:
    win = sum(1 for x in xs if x > 0) / len(xs)
    print(f'  ST组 n={len(xs)} 均值 {st.mean(xs)*100:+.1f}% 中位 {q(xs,0.5)*100:+.1f}% 胜率 {win*100:.1f}%')
    print(f'  分位: 10% {q(xs,0.1)*100:+.1f}% | 50% {q(xs,0.5)*100:+.1f}% | 90% {q(xs,0.9)*100:+.1f}%')
xs_all = window_buy(panel, '2025-10-15', '2025-11-07')
if xs_all:
    win = sum(1 for x in xs_all if x > 0) / len(xs_all)
    print(f'  全市场 n={len(xs_all)} 均值 {st.mean(xs_all)*100:+.1f}% 中位 {q(xs_all,0.5)*100:+.1f}% 胜率 {win*100:.1f}%')
