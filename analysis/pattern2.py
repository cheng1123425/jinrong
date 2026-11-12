# -*- coding: utf-8 -*-
import pickle, statistics as st
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = pickle.load(open(A + '/panel2425.pkl','rb'))

def q(xs,p):
    xs=sorted(xs)
    if not xs: return float('nan')
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)
def ma(c,i,n=60):
    return sum(c[i-n+1:i+1])/n if i>=n-1 else None
def st_events(panel,year):
    res={}
    for code,d in panel.items():
        idx=[i for i,dt in enumerate(d['dates']) if dt[:4]==str(year)]
        if not idx or 'ST' in d['name'][idx[0]]: continue
        prev=False
        for i in idx:
            cur='ST' in d['name'][i]
            if cur and not prev: res[code]=i; break
            prev=cur
    return res

def pat2(d, year, bounce=0.12, hold=20):
    """突破MA60 -> 5日内涨停确认 -> 买点=涨停日收盘"""
    dates,c,l,h,pc=d['dates'],d['c'],d['l'],d['h'],d['pc']
    idx=[i for i,dt in enumerate(dates) if dt[:4]==str(year)]
    if len(idx)<140: return None
    win=[i for i in idx if f'{year}-05-01'<=dates[i]<=f'{year}-06-30']
    if not win: return None
    i_low=min(win,key=lambda i:l[i]); low=l[i_low]
    after=[i for i in idx if i>i_low]
    if not after: return None
    i_high=max(after,key=lambda i:h[i])
    if h[i_high]/low-1 < bounce: return None
    pull=[i for i in after if i>i_high and dates[i]<f'{year}-10-01']
    if not pull: return None
    i_pull=min(pull,key=lambda i:l[i])
    if l[i_pull] <= low*1.005: return None      # 不创新低
    # 10月首次突破 MA60
    b1=None
    for i in idx:
        if not (f'{year}-10-01'<=dates[i]<=f'{year}-11-10'): continue
        m,mp=ma(c,i),ma(c,i-1)
        if m and mp and c[i]>m and c[i-1]<=mp: b1=i; break
    if b1 is None: return None
    # 突破后 5 日内涨停(>=4.8%)确认
    for i in range(b1, min(b1+6, len(c))):
        if pc[i]>0 and c[i]/pc[i]-1 >= 0.048:
            if i+hold < len(c):
                return b1, i, c[i+hold]/c[i]-1, dates[i]
            return None
    return None

for year in (2024,2025):
    ev=st_events(panel,year); hits=[]
    for code,i in ev.items():
        r=pat2(panel[code],year)
        if r: hits.append((code,)+r)
    print(f'\n===== {year} 新增ST {len(ev)} 只 | 突破+涨停确认命中 {len(hits)} 只 =====')
    if hits:
        rets=[x[3] for x in hits]
        print(f'  买点后20日: 中位 {q(rets,0.5)*100:+.1f}% 均值 {st.mean(rets)*100:+.1f}% '
              f'胜率 {sum(1 for r in rets if r>0)/len(rets)*100:.1f}% P10 {q(rets,0.1)*100:+.1f}% P90 {q(rets,0.9)*100:+.1f}%')
        print(f'  涨超20%: {sum(1 for r in rets if r>0.2)}/{len(rets)} | 跌超10%: {sum(1 for r in rets if r<-0.1)}/{len(rets)}')
        for code,b1,i,ret,dt in sorted(hits,key=lambda x:-x[3])[:10]:
            print(f'    {code} 突破{panel[code]["dates"][b1]} 信号{dt} 收益{ret*100:+.1f}%')
    else:
        print('  无样本')
