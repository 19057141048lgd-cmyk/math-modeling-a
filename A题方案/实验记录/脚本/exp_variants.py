"""各策略开关的官方对比：python exp_variants.py <问题> <核数> <用例...>"""
import csv
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, '.')
from solver.config import case_path, load_config
from solver.evaluate import run_official
from solver.graph import Graph
from solver.solve import build_candidate

VARIANTS = {
    1: [('A', 0.25, 16, 2.0), ('A', 0.25, 16, 2.0, None, 'child'), ('A', 0.25, 16, 2.0, None, 'blc'),
        ('A', 0.1, 16, 2.0, 4), ('A', 0.1, 16, 2.0, 4, 'child'), ('A', 0.25, 16, 2.0, None, 'mc'),
        ('A', 0.25, 16, 2.0, 16, 'mc'), ('A', 0.1, 16, 2.0, 4, 'mc')],
    2: [('B', 0.1, 16, 8.0), ('B', 0.1, 16, 8.0, None, 'mc'), ('B', 0.1, 16, 8.0, None, 'mr'),
        ('B', 0.1, 16, 8.0, None, 'mc+mr'), ('B', 0.1, 16, 8.0, None, 'mc+mr+blc'),
        ('B', 0.1, 64, 2.0, None, 'mc+mr'), ('B', 0.1, 16, 8.0, None, 'child')],
    3: [('C', 0.1, 64, 4.0), ('C', 0.1, 64, 4.0, None, 'mc+mr'), ('C', 0.1, 64, 4.0, None, 'mc+mr+blc'),
        ('C', 0.25, 16, 8.0, None, 'mc+mr'), ('C', 0.1, 64, 4.0, None, 'child')],
}
hw = load_config()
q, n = int(sys.argv[1]), int(sys.argv[2])
only = os.environ.get('ONLY')
rows = {(r['case'], r['problem'], r['cores']): r
        for r in csv.DictReader(open('results_all/summary.csv', encoding='utf-8-sig'))}
for case in sys.argv[3:]:
    G = Graph(case_path(case))
    cur = rows[(case, str(q), str(n))]
    print('%s P%d N=%d 现有选择: makespan=%s spill=%s params=%s' % (
        case, q, n, cur['makespan'], cur['spill_added_copy_bytes'], cur['params']), flush=True)
    for params in VARIANTS[q]:
        if only and only not in str(params):
            continue
        t0 = time.time()
        plan, pred, n_sub, n_cl, spill = build_candidate(G, n, hw, *params)
        dt = time.time() - t0
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'plan.json')
            json.dump(plan, open(p, 'w'))
            out = run_official(case, q, p, tmp, tag='x')
        print('   %-40s 簇=%5d 子图=%5d 预测=%9.0f spill预测=%10d | 官方=%9d spill=%10d (建 %.1fs)' % (
            params, n_cl, n_sub, pred, spill, out['makespan'], out.get('spill_added_copy_bytes', -1), dt), flush=True)
