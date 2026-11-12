# -*- coding: utf-8 -*-
import json, os, sys, statistics as st
sys.path.insert(0, r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis')
from pattern_similarity import read60, scan

SRC = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/tmp60/60分钟'
A = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
ev = json.load(open(A + '/st_events.json', encoding='utf-8'))
new_st = {k: v for k, v in ev.items() if v['st_date']}

def q(xs,p):
    xs=sorted(xs)
    if not xs: return float('nan')
    k=(len(xs)-1)*p; f=int(k); c=min(f+1,len(xs)-1)
    return xs[f]+(xs[c]-xs[f])*(k-f)

def ret_after(closes, i, hold=80):
    j = min(i + hold, len(closes) - 1)
    return closes[j] / closes[i] - 1

rows = []
n_sig = 0
for code in new_st:
    p = os.path.join(SRC, code + '.csv')
    if not os.path.exists(p): continue
    bars, closes = read60(p)
    if len(bars) < 300: continue
    sigs = scan(bars, closes, step=4)
    n_sig += len(sigs)
    # 找 10/15-11/07 的信号
    win = [s for s in sigs if '2025-10-15' <= s['date'] <= '2025-11-07']
    if win:
        s = win[0]
        i = next(i for i, b in enumerate(bars) if b[0] == s['date'] and b[1] == s['time'])
        rows.append((code, s['date'], s['score'], ret_after(closes, i)))

print(f'ST 样本: {len(new_st)} | 总信号: {n_sig} | 平均每只 {n_sig/max(len(new_st),1):.1f} 个')
print(f'10/15-11/07 有信号的股票: {len(rows)} 只 ({len(rows)/len(new_st)*100:.1f}%)')
if rows:
    rets = [x[3] for x in rows]
    print(f'  信号后 20 交易日: 中位 {q(rets,0.5)*100:+.1f}% 均值 {st.mean(rets)*100:+.1f}% '
          f'胜率 {sum(1 for r in rets if r>0)/len(rets)*100:.0f}% P90 {q(rets,0.9)*100:+.1f}%')
    print('  前 10 个:')
    for code, dt, sc, r in sorted(rows, key=lambda x: -x[3])[:10]:
        print(f'    {code} {dt} 分数{sc:.3f} 收益{r*100:+.1f}%')
else:
    print('  无信号 → 按规范的判据（信号<10）直接判 «证据不足»')
