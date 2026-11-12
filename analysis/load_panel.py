# -*- coding: utf-8 -*-
import zipfile, pickle, os
ZIP = r'E:/全量历史数据-/stock-trading-data-2025-12-23N.zip'
OUT = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
z = zipfile.ZipFile(ZIP)
panel = {}
bad = 0
for i, n in enumerate([x for x in z.namelist() if x.endswith('.csv')]):
    raw = z.read(n).decode('gbk', 'replace')
    lines = raw.splitlines()
    if len(lines) < 3:
        continue
    rows = [l.split(',') for l in lines[2:]]
    rows = [p for p in rows if len(p) >= 12 and p[2] >= '2025-01-01']
    if not rows:
        continue
    code = rows[0][0]
    d = {'dates': [], 'name': [], 'o': [], 'h': [], 'l': [], 'c': [], 'pc': [], 'v': []}
    for p in rows:
        try:
            o, h, l, c, pc, v = (float(p[3]), float(p[4]), float(p[5]),
                                 float(p[6]), float(p[7]), float(p[8]))
        except Exception:
            bad += 1
            continue
        d['dates'].append(p[2]); d['name'].append(p[1])
        d['o'].append(o); d['h'].append(h); d['l'].append(l)
        d['c'].append(c); d['pc'].append(pc); d['v'].append(v)
    if len(d['dates']) >= 20:
        panel[code] = d
    if i % 1500 == 0:
        print('  loaded', i, 'kept', len(panel))
pickle.dump(panel, open(os.path.join(OUT, 'panel2025.pkl'), 'wb'), protocol=4)
print('stocks with >=20 obs:', len(panel), '| bad rows:', bad)
sizes = sorted(len(v['dates']) for v in panel.values())
print('rows per stock: min', sizes[0], 'median', sizes[len(sizes)//2], 'max', sizes[-1])
