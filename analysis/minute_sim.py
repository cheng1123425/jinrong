# -*- coding: utf-8 -*-
import csv, os, json, statistics as st
import numpy as np
SRC = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/tmp60/60分钟'
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
ev = json.load(open(A + '/st_events.json', encoding='utf-8'))
new_st = {k: v for k, v in ev.items() if v['st_date']}

def load60(code):
    p = os.path.join(SRC, code + '.csv')
    if not os.path.exists(p): return None
    rows = []
    with open(p, encoding='utf-8-sig') as f:
        rd = csv.reader(f); next(rd, None)
        for r in rd:
            if len(r) < 7: continue
            try: rows.append((r[0], float(r[2]), float(r[3]), float(r[4]), float(r[5])))
            except ValueError: pass
    return rows

def q(xs,p):
    xs=sorted(xs)
    if not xs: return float('nan')
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

# 在 10/15-11/07 首个交易日买入，比较前 80 根 60min 的「前40 vs 后40」收益相关性
res = []
for code in new_st:
    rows = load60(code)
    if not rows or len(rows) < 200: continue
    idx = next((i for i, r in enumerate(rows) if '2025-10-15' <= r[0] <= '2025-11-07'), None)
    if idx is None or idx < 85 or idx + 20*4 >= len(rows): continue
    closes = np.array([r[4] for r in rows[:idx]])
    r = np.diff(np.log(closes))[-80:]
    a, b = r[:40], r[40:]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: continue
    corr = float(np.corrcoef(a, b)[0, 1])
    buy = rows[idx][4]
    sell = rows[min(idx + 20*4, len(rows)-1)][4]
    res.append((code, corr, sell/buy - 1))

print(f'样本: {len(res)} 只 ST（10/15-11/07 买入，持有 20 个交易日）')
if res:
    corrs = [x[1] for x in res]
    print(f'60分钟收益自相关（前40 vs 后40）: 中位 {q(corrs,0.5):+.3f} | P10 {q(corrs,0.1):+.3f} | P90 {q(corrs,0.9):+.3f}')
    hi = [x[2] for x in res if x[1] > 0.5]
    lo = [x[2] for x in res if x[1] <= 0.5]
    for nm, xs in (('高自相似(>0.5)', hi), ('低自相似(<=0.5)', lo)):
        if xs:
            print(f'  {nm}: n={len(xs)} 中位 {q(xs,0.5)*100:+.1f}% 均值 {st.mean(xs)*100:+.1f}% 胜率 {sum(1 for x in xs if x>0)/len(xs)*100:.0f}%')
        else:
            print(f'  {nm}: 无样本')
    # 相关性 vs 收益的秩相关
    if len(res) > 10:
        from scipy.stats import spearmanr
        rho, p = spearmanr([x[1] for x in res], [x[2] for x in res])
        print(f'  自相似度与后续收益的 Spearman: {rho:+.3f} (p={p:.3f})')
