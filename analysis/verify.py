# -*- coding: utf-8 -*-
import pickle, statistics as st, collections
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = pickle.load(open(A + '/panel2425.pkl', 'rb'))

def q(xs, p):
    xs = sorted(xs)
    if not xs: return float('nan')
    k = (len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def st_events(panel, year):
    res = {}
    for code, d in panel.items():
        idx = [i for i, dt in enumerate(d['dates']) if dt[:4] == str(year)]
        if not idx: continue
        f = idx[0]
        if 'ST' in d['name'][f]: continue          # 年初已ST，不计新增
        prev = False
        for i in idx:
            cur = 'ST' in d['name'][i]
            if cur and not prev:
                res[code] = i
                break
            prev = cur
    return res

def summarize(name, xs):
    if not xs: print(f'  {name}: 无样本'); return
    print(f'  {name:<10} n={len(xs):3d} 中位 {q(xs,0.5)*100:+6.1f}% 均值 {st.mean(xs)*100:+6.1f}% '
          f'胜率 {sum(1 for x in xs if x>0)/len(xs)*100:5.1f}% P10 {q(xs,0.1)*100:+6.1f}% P90 {q(xs,0.9)*100:+6.1f}%')

for year in (2024, 2025):
    ev = st_events(panel, year)
    print(f'\n########## {year} 年新增 ST：{len(ev)} 只 ##########')
    mon = collections.Counter(panel[c]['dates'][i][:7] for c, i in ev.items())
    print('  戴帽月份:', dict(sorted(mon.items())))

    # 戴帽当天
    d0 = []
    for c, i in ev.items():
        d = panel[c]
        if d['pc'][i] > 0: d0.append(d['c'][i]/d['pc'][i]-1)
    summarize('戴帽当天', d0)

    # 当年 10/31 与 12/31 收益（从戴帽日收盘）
    end31, endyr = [], []
    for c, i in ev.items():
        d = panel[c]
        j31 = [j for j, dt in enumerate(d['dates']) if dt[:4]==str(year) and dt <= f'{year}-10-31']
        jye = [j for j, dt in enumerate(d['dates']) if dt[:4]==str(year)]
        if j31 and j31[-1] > i: end31.append(d['c'][j31[-1]]/d['c'][i]-1)
        if jye and jye[-1] > i: endyr.append(d['c'][jye[-1]]/d['c'][i]-1)
    summarize('->10/31', end31)
    summarize('->年末', endyr)

    # 见底月份（5/1 之后）
    lm = collections.Counter()
    lows = []
    for c, i in ev.items():
        d = panel[c]
        idx = [j for j, dt in enumerate(d['dates']) if dt >= f'{year}-05-01' and dt[:4]==str(year)]
        if not idx: continue
        j = min(idx, key=lambda k: d['l'][k])
        lm[d['dates'][j][:7]] += 1
        lows.append((c, d['dates'][j]))
    print('  5/1后最低点月份:', dict(sorted(lm.items())))
    jun = sum(1 for _, dt in lows if dt[5:7]=='06')
    print(f'  6月见底占比: {jun}/{len(lows)} = {jun/len(lows)*100:.1f}%')

    # 10/15-11/07 买入持有20日
    def buy(panel_sub, lo, hi, hold=20):
        out=[]
        for c,d in panel_sub.items():
            cand=[i for i,dt in enumerate(d['dates']) if lo<=dt<=hi]
            if not cand: continue
            i=cand[0]
            if i+hold < len(d['c']): out.append((d['c'][i+hold]/d['c'][i]-1, d['fmv'][i]))
        return out
    stsub = {c: panel[c] for c in ev}
    s = buy(stsub, f'{year}-10-15', f'{year}-11-07')
    allb = buy(panel, f'{year}-10-15', f'{year}-11-07')
    print(f'  10/15-11/07 买入持有20日:')
    summarize('ST组', [x for x,_ in s])
    summarize('全市场', [x for x,_ in allb])
    if allb:
        mv = sorted(allb, key=lambda t: t[1])
        n20 = max(1, len(mv)//5)
        summarize('小市值20%', [x for x,_ in mv[:n20]])
        # 同市值域对照：把全市场按市值分组，看 ST 落在哪
        st_mv = sorted(x for _,x in s)
        print(f'    ST组流通市值中位: {q(st_mv,0.5)/1e8:.1f}亿 | 全市场中位: {q([x for _,x in allb],0.5)/1e8:.1f}亿')
