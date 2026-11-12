# -*- coding: utf-8 -*-
import zipfile, pickle, os
ZIP = r'E:/全量历史数据-/stock-trading-data-2025-12-23N.zip'
OUT = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
z = zipfile.ZipFile(ZIP)
panel = {}
for i, n in enumerate([x for x in z.namelist() if x.endswith('.csv')]):
    raw = z.read(n).decode('gbk', 'replace')
    lines = raw.splitlines()
    if len(lines) < 3: continue
    rows = [l.split(',') for l in lines[2:]]
    rows = [p for p in rows if len(p) >= 12 and p[2] >= '2024-01-01']
    if not rows: continue
    d = {'dates': [], 'name': [], 'o': [], 'h': [], 'l': [], 'c': [], 'pc': [], 'v': [], 'fmv': []}
    for p in rows:
        try:
            o,h,l,c,pc,v,fmv = float(p[3]),float(p[4]),float(p[5]),float(p[6]),float(p[7]),float(p[8]),float(p[10])
        except Exception: continue
        d['dates'].append(p[2]); d['name'].append(p[1])
        d['o'].append(o); d['h'].append(h); d['l'].append(l); d['c'].append(c)
        d['pc'].append(pc); d['v'].append(v); d['fmv'].append(fmv)
    if len(d['dates']) >= 40: panel[rows[0][0]] = d
    if i % 2000 == 0: print('  ', i, 'kept', len(panel))
pickle.dump(panel, open(os.path.join(OUT,'panel2425.pkl'),'wb'), protocol=4)
print('stocks:', len(panel))
print('date range sample:', panel[list(panel)[0]]['dates'][0], '->', panel[list(panel)[0]]['dates'][-1])
