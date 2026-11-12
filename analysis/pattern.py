# -*- coding: utf-8 -*-
import pickle, statistics as st, collections
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = pickle.load(open(A + '/panel2425.pkl', 'rb'))

def q(xs,p):
    xs=sorted(xs)
    if not xs: return float('nan')
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def ma(c, i, n=60):
    if i < n-1: return None
    return sum(c[i-n+1:i+1])/n

def st_events(panel, year):
    res={}
    for code,d in panel.items():
        idx=[i for i,dt in enumerate(d['dates']) if dt[:4]==str(year)]
        if not idx: continue
        if 'ST' in d['name'][idx[0]]: continue
        prev=False
        for i in idx:
            cur='ST' in d['name'][i]
            if cur and not prev: res[code]=i; break
            prev=cur
    return res

def pattern(d, year, bounce=0.15, pullback=0.06, near_ma=0.06, min_near=5, hold=20):
    """N字上涨->N字回落不创新低->60日线附近磨->10月突破MA60。返回 (买点索引, 收益) 或 None"""
    dates, c, l, h = d['dates'], d['c'], d['l'], d['h']
    idx=[i for i,dt in enumerate(dates) if dt[:4]==str(year)]
    if len(idx) < 140: return None
    win=[i for i in idx if f'{year}-05-01' <= dates[i] <= f'{year}-06-30']
    if not win: return None
    i_low=min(win, key=lambda i: l[i]); low=l[i_low]
    after=[i for i in idx if i > i_low]
    if not after: return None
    i_high=max(after, key=lambda i: h[i])
    if h[i_high] < low*(1+bounce): return None            # N字上涨不足
    if h[i_high]/low - 1 < bounce: return None
    pull=[i for i in after if i > i_high and dates[i] < f'{year}-10-01']
    if not pull: return None
    i_pull=min(pull, key=lambda i: l[i])
    if l[i_pull] <= low*1.005: return None                 # 创新低，淘汰
    if h[i_high]/l[i_pull] - 1 < pullback: return None     # 回落幅度不足，不成N字
    # 60日线附近磨：i_pull 前后 15 日内，收盘距 MA60 在 near_ma 内的天数
    near=0
    for i in range(max(0,i_pull-15), min(len(c), i_pull+16)):
        m=ma(c,i)
        if m and abs(c[i]/m-1) <= near_ma: near+=1
    if near < min_near: return None
    # 10/01-11/15 突破 MA60
    brk=None
    for i in idx:
        if not (f'{year}-10-01' <= dates[i] <= f'{year}-11-15'): continue
        m=ma(c,i); mprev=ma(c,i-1)
        if m and mprev and c[i] > m and c[i-1] <= mprev:
            brk=i; break
    if brk is None or brk+hold >= len(c): return None
    return brk, c[brk+hold]/c[brk]-1, dates[brk]

for year in (2024,2025):
    ev=st_events(panel,year)
    hits=[]; nobreak=0
    for code,i in ev.items():
        r=pattern(panel[code],year)
        if r: hits.append((code,)+r)
        else: nobreak+=1
    print(f'\n===== {year} 新增ST {len(ev)} 只 =====')
    print(f'  符合形态并出现10月突破: {len(hits)} 只 ({len(hits)/len(ev)*100:.1f}%)')
    if hits:
        rets=[x[2] for x in hits]
        print(f'  买点后20日: 中位 {q(rets,0.5)*100:+.1f}% 均值 {st.mean(rets)*100:+.1f}% '
              f'胜率 {sum(1 for r in rets if r>0)/len(rets)*100:.1f}% P10 {q(rets,0.1)*100:+.1f}% P90 {q(rets,0.9)*100:+.1f}%')
        print('  样本:')
        for code,brk,ret,dt in sorted(hits,key=lambda x:-x[2])[:15]:
            print(f'    {code} 买点{dt} 收益{ret*100:+.1f}%')
    else:
        print('  无样本，需放宽条件')
