# -*- coding: utf-8 -*-
import csv, os, pickle
SRC = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/tmp60/60分钟'
OUT = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
panel = {}
files = [f for f in os.listdir(SRC) if f.endswith('.csv')]
for i, fn in enumerate(files):
    code = fn[:-4]
    dates, O, H, L, C, V = [], [], [], [], [], []
    cur = None
    o = h = l = c = v = 0.0
    try:
        with open(os.path.join(SRC, fn), encoding='utf-8-sig') as f:
            rd = csv.reader(f)
            next(rd, None)
            for row in rd:
                if len(row) < 7: continue
                d = row[0]
                try:
                    oo, hh, ll, cc, vv = float(row[2]), float(row[3]), float(row[4]), float(row[5]), float(row[6])
                except ValueError:
                    continue
                if d != cur:
                    if cur is not None:
                        dates.append(cur); O.append(o); H.append(h); L.append(l); C.append(c); V.append(v)
                    cur = d; o, h, l, c, v = oo, hh, ll, cc, vv
                else:
                    h = max(h, hh); l = min(l, ll); c = cc; v += vv
            if cur is not None:
                dates.append(cur); O.append(o); H.append(h); L.append(l); C.append(c); V.append(v)
    except Exception as e:
        continue
    if len(dates) >= 40:
        panel[code] = {'dates': dates, 'o': O, 'h': H, 'l': L, 'c': C, 'v': V}
    if i % 1500 == 0: print('  ', i, 'kept', len(panel))
pickle.dump(panel, open(os.path.join(OUT, 'panel60_daily.pkl'), 'wb'), protocol=4)
print('stocks:', len(panel))
k = list(panel)[0]
print('range:', panel[k]['dates'][0], '->', panel[k]['dates'][-1], '| days:', len(panel[k]['dates']))
