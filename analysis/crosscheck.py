# -*- coding: utf-8 -*-
import pickle, json, statistics as st, collections
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
p60 = pickle.load(open(A + '/panel60_daily.pkl', 'rb'))
pday = pickle.load(open(A + '/panel2425.pkl', 'rb'))
ev = json.load(open(A + '/st_events.json', encoding='utf-8'))

def q(xs,p):
    xs=sorted(xs)
    if not xs: return float('nan')
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

# 1) 数据一致性：同 code 的 2025 收盘价
diffs=[]
common=[c for c in p60 if c in pday][:400]
for c in common:
    d60=dict(zip(p60[c]['dates'], p60[c]['c']))
    for dt,cl in zip(pday[c]['dates'], pday[c]['c']):
        if dt in d60 and dt>='2025-01-01':
            diffs.append(abs(d60[dt]/cl-1) if cl else 0)
if diffs:
    print(f'一致性检查 n={len(diffs)} 收盘价相对偏差: 中位 {q(diffs,0.5)*100:.4f}% | 90% {q(diffs,0.9)*100:.3f}% | 最大 {max(diffs)*100:.2f}%')

new_st={k:v for k,v in ev.items() if v['st_date'] and k in p60}
print(f'2025 新增 ST 且在 60min 库: {len(new_st)}')

# 2) 戴帽当天（用前一日收盘）
d0=[]
for c,v in new_st.items():
    d=p60[c]
    try: i=d['dates'].index(v['st_date'])
    except ValueError: continue
    if i>0 and d['c'][i-1]>0: d0.append(d['c'][i]/d['c'][i-1]-1)
print(f'\n=== 戴帽当天（60min 聚合）n={len(d0)} ===')
print(f'  中位 {q(d0,0.5)*100:+.2f}% 均值 {st.mean(d0)*100:+.2f}% 胜率 {sum(1 for x in d0 if x>0)/len(d0)*100:.1f}%')

# 3) 5/1 后见底月份
lm=collections.Counter()
for c,v in new_st.items():
    d=p60[c]
    idx=[i for i,dt in enumerate(d['dates']) if dt>='2025-05-01']
    if not idx: continue
    j=min(idx, key=lambda i:d['l'][i]); lm[d['dates'][j][:7]]+=1
print('\n=== 5/1 后最低点月份（60min 聚合）===')
for k in sorted(lm): print(f'  {k}: {lm[k]}')
tot=sum(lm.values()); jun=lm.get('2025-06',0)
print(f'  6月占比: {jun}/{tot} = {jun/tot*100:.1f}%')

# 4) 10/15-11/07 买入持有20日
def buy(pan, lo='2025-10-15', hi='2025-11-07', hold=20, names=None):
    out=[]
    for c,d in pan.items():
        if names and c not in names: continue
        cand=[i for i,dt in enumerate(d['dates']) if lo<=dt<=hi]
        if not cand: continue
        i=cand[0]
        if i+hold<len(d['c']): out.append(d['c'][i+hold]/d['c'][i]-1)
    return out
s=buy(p60, names=set(new_st))
a=buy(p60)
print(f'\n=== 10/15-11/07 买入持有20日（60min 聚合）===')
for nm,xs in (('ST组',s),('全市场',a)):
    if xs:
        print(f'  {nm} n={len(xs)} 中位 {q(xs,0.5)*100:+.1f}% 均值 {st.mean(xs)*100:+.1f}% 胜率 {sum(1 for x in xs if x>0)/len(xs)*100:.1f}% P90 {q(xs,0.9)*100:+.1f}%')
