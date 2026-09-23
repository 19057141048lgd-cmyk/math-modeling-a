"""开发期模型校准：1 核预测 vs 官方单核 makespan。"""
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

from .config import SOLUTION_DIR, case_path, load_config
from .evaluate import single_core_makespan
from .graph import Graph, build_clusters
from .sched import schedule

OUT = os.path.join(SOLUTION_DIR, 'results', 'dev')


def one(case):
    G = Graph(case_path(case))
    clusters, _ = build_clusters(G, 1, math.inf, 16)
    S = schedule(G, clusters, 1, 'B', load_config())
    return case, S.makespan, single_core_makespan(case, OUT), G.total_cycles


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    os.makedirs(OUT, exist_ok=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        for case, pred, real, tot in ex.map(one, sys.argv[1:]):
            print('%s  model=%.0f  official=%d  err=%+.1f%%  sum_cycles=%d' % (
                case, pred, real, 100.0 * (pred - real) / real, tot), flush=True)
