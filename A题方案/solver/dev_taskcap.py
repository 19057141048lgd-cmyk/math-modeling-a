"""开发期：场景 A 的 Task 规模上限扫描。"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import run_official, single_core_makespan
from .graph import Graph
from .solve import build_candidate

OUT = os.path.join(SOLUTION_DIR, 'results', 'dev', 'taskcap')


def one(job):
    case, split, div = job
    G = Graph(case_path(case))
    plan, pred, n_sub, _, _ = build_candidate(G, 4, load_config(), 'A', split, 16, 2.0, div)
    tag = 'cap_%s_%s' % (split, div)
    path = os.path.join(OUT, '%s_%s.json' % (case, tag))
    json.dump(plan, open(path, 'w', encoding='utf-8'))
    return job, n_sub, run_official(case, 1, path, OUT, tag=tag)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUT, exist_ok=True)
    cases = sys.argv[1:]
    divs = [None, 4, 8, 16, 32, 64, 128]
    jobs = [(c, s, d) for c in cases for s in (0.1, 0.25) for d in divs]
    with ProcessPoolExecutor(max_workers=8) as ex:
        singles = dict(zip(cases, ex.map(single_core_makespan, cases,
                                         [os.path.join(SOLUTION_DIR, 'results', 'single')] * len(cases))))
        res = list(ex.map(one, jobs))
    for case in cases:
        line = case
        for (c, s, d), n_sub, r in res:
            if c == case:
                line += '  %s/%s:%.2f(sg%d,spill%dK)' % (s, d, singles[c] / r['makespan'], n_sub,
                                                         r['spill_added_copy_bytes'] // 1000)
        print(line)
