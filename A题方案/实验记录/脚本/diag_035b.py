import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
out = os.path.join(os.environ['TEMP'], 'mm_extract', 'diag035')
case = os.environ.get('CASE', 'case_035')
main = json.load(open(os.path.join(out, '%s_d.json' % case), encoding='utf-8'))
tl = main['per_core_timeline']
ev = []
for core in tl:
    for e in core['ops']:
        ev.append((e['start'], e['end'], core['core_id'], e))
# 全局空闲区间：所有核所有 pipe 都不忙
ev.sort(key=lambda x: x[0])
gaps, reach = [], 0
for s, e, k, op in ev:
    if s > reach + 500:
        gaps.append((reach, s))
    reach = max(reach, e)
print('全局空闲 >500 周期的区间：', [(int(a), int(b), int(b - a)) for a, b in gaps])
# 各核最长的空闲区间，以及空闲结束时发射的算子
for core in tl:
    ops = sorted(core['ops'], key=lambda e: e['start'])
    reach, best = 0, []
    for e in ops:
        if e['start'] - reach > 2000:
            best.append((e['start'] - reach, reach, e))
        reach = max(reach, e['end'])
    best.sort(key=lambda x: -x[0])
    for gap, r, e in best[:3]:
        print('核%d 空闲 %d（%d -> %d），之后首个算子 %s' % (core['core_id'], gap, r, e['start'],
                                                     {k: e[k] for k in ('op_id', 'op', 'pipe', 'duration', 'subgraph_id')}))
# 最长的算子
ops = sorted(ev, key=lambda x: -(x[1] - x[0]))[:8]
for s, e, k, op in ops:
    print('长算子 核%d %s %s dur=%d start=%d sg=%s' % (k, op['op'], op['pipe'], e - s, s, op.get('subgraph_id')))
xf = main.get('cross_core_transfers')
print(type(xf), (list(xf[0]) if isinstance(xf, list) and xf else xf if not isinstance(xf, list) else None))
