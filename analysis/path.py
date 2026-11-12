# -*- coding: utf-8 -*-
import pickle, json, statistics as st
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = pickle.load(open(A + '/panel2025.pkl', 'rb'))
ev = json.load(open(A + '/st_events.json', encoding='utf-8'))
new_st = {k: v for k, v in ev.items() if v['st_date'] and k in panel}

# 戴帽日当天与次日
day0, d1, d2, d3 = [], [], [], []
for code, v in new_st.items():
    d = panel[code]
    try: i = d['dates'].index(v['st_date'])
    except ValueError: continue
    if d['pc'][i] > 0: day0.append(d['c'][i] / d['pc'][i] - 1)
    for arr, T in ((d1, 1), (d2, 2), (d3, 3)):
        if i + T < len(d['c']): arr.append(d['c'][i + T] / d['c'][i] - 1)

def show(name, xs):
    if not xs: return
    print(f'  {name:<12} n={len(xs):3d} 均值 {st.mean(xs)*100:+6.2f}% 中位 {sorted(xs)[len(xs)//2]*100:+6.2f}% '
          f'胜率 {sum(1 for x in xs if x>0)/len(xs)*100:5.1f}% 最差 {min(xs)*100:+6.1f}% 最好 {max(xs)*100:+6.1f}%')

print('=== 戴帽时点收益（相对戴帽日收盘）===')
show('当天', day0); show('T+1', d1); show('T+2', d2); show('T+3', d3)

# 等权平均累计路径（以戴帽日为锚）
print('\n=== 等权平均累计收益路径（戴帽日=0）===')
for T in (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120):
    xs = []
    for code, v in new_st.items():
        d = panel[code]
        try: i = d['dates'].index(v['st_date'])
        except ValueError: continue
        if i + T < len(d['c']) and d['c'][i] > 0:
            xs.append(d['c'][i + T] / d['c'][i] - 1)
    if xs:
        med = sorted(xs)[len(xs)//2]
        print(f'  T+{T:<3d} n={len(xs):3d} 均值 {st.mean(xs)*100:+7.2f}%  中位 {med*100:+7.2f}%')

# 按月：从戴帽日到每月末的收益
print('\n=== 戴帽后到各月末的收益 ===')
import collections
for m in ['2025-05','2025-06','2025-07','2025-08','2025-09','2025-10','2025-11','2025-12']:
    xs = []
    for code, v in new_st.items():
        d = panel[code]
        try: i = d['dates'].index(v['st_date'])
        except ValueError: continue
        idx = [j for j, dt in enumerate(d['dates']) if dt[:7] == m]
        if not idx: continue
        j = idx[-1]
        if j > i: xs.append(d['c'][j] / d['c'][i] - 1)
    if xs:
        med = sorted(xs)[len(xs)//2]
        print(f'  ->{m} n={len(xs):3d} 均值 {st.mean(xs)*100:+7.2f}% 中位 {med*100:+7.2f}% 胜率 {sum(1 for x in xs if x>0)/len(xs)*100:5.1f}%')
