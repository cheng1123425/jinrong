# -*- coding: utf-8 -*-
import zipfile, json, collections, os
ZIP = r'E:/全量历史数据-/stock-trading-data-2025-12-23N.zip'
OUT = r'C:/Users/Administrator/WorkBuddy/2026-11-09-23-53-01/analysis'
z = zipfile.ZipFile(ZIP)
names = [n for n in z.namelist() if n.endswith('.csv')]
print('files:', len(names))
recs = {}
for i, n in enumerate(names):
    raw = z.read(n).decode('gbk', 'replace')
    lines = raw.splitlines()
    if len(lines) < 3:
        continue
    rows = [l.split(',') for l in lines[2:]]
    rows = [p for p in rows if len(p) >= 12 and p[2] >= '2025-01-01']
    if not rows:
        continue
    code = rows[0][0]
    first_name = rows[0][1]
    st_start = 'ST' in first_name
    new_st_date, new_st_name = None, None
    prev = st_start
    for p in rows:
        cur = 'ST' in p[1]
        if cur and not prev:
            new_st_date, new_st_name = p[2], p[1]
            break
        prev = cur
    recs[code] = dict(first=first_name, st_start=st_start, st_date=new_st_date,
                      st_name=new_st_name, n=len(rows))
    if i % 1500 == 0:
        print('  scanned', i)
json.dump(recs, open(os.path.join(OUT, 'st_events.json'), 'w', encoding='utf-8'), ensure_ascii=False)
print('total stocks:', len(recs))
print('ST at 2025 start:', sum(1 for v in recs.values() if v['st_start']))
newst = {k: v for k, v in recs.items() if v['st_date']}
print('new ST in 2025:', len(newst))
mon = collections.Counter(v['st_date'][:7] for v in newst.values())
for k in sorted(mon): print('  ', k, mon[k])
day = collections.Counter(v['st_date'] for v in newst.values())
print('top dates:', day.most_common(12))
