"""为单个用例生成问题 1/2/3 的多核切分与调度方案。

流程：少量固定候选参数 -> 两级粗化 + 通信感知列表调度 -> 代理模型预测 makespan 排序
      -> （可选）前 K 名交官方评估器复核取最优。

用法（在 A题方案 目录下）：
    python -m solver.solve case_019 --cores 4 --problem 2 -o results/case_019_p2_n4.json --verify 3
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time

from .config import case_path, load_config
from .graph import Graph, build_clusters, cluster_parallelism, single_cluster
from .memory import estimate_spill
from .plan import plan_scene_a, plan_scene_b, validate_plan
from .sched import schedule

INF = math.inf
# spill 流量经过共享 DDR，按 SPILL_ALPHA * u * 字节 / 带宽 计入 makespan，u 为 DDR 利用率：
# DDR 越空闲，spill 越能被计算掩盖。16 个代表用例上标定，另 10 个用例（含 4 个大图）上验证（dev_spill_time.py）。
SPILL_ALPHA = 1.2
# 内存感知聚簇 / 驻留跟踪只给已搬入的图输入和跨核张量留出片上容量的这一比例，其余留给中间结果。
MEM_FRAC = 0.5

# (调度场景, split_ratio, grain, locality[, task_div])，由 10 个代表用例的参数扫描确定；
# split_ratio=inf 即“独立作业整体分配”的朴素基线，保证不劣于基线；
# task_div：场景 A 单个 Task 计算量上限 W/(N*task_div)，用于限制工作集、消除 spill。
# 第 6 项为策略开关（见 build_candidate），挂在各问题最常被选中的参数上。
# child 与 mr 不进候选池：23 个用例上 child 净效果为零（5 次选中 2 次退化），mr 与 spill 预测重复计费。
CANDIDATES = {
    1: [('A', 0.25, 16, 2.0), ('A', 0.25, 16, 1.0), ('A', 1.0, 16, 2.0), ('A', 0.25, 16, 4.0),
        ('A', 1.0, 64, 8.0), ('A', 0.1, 64, 1.0), ('A', 1.0, 16, 0.0), ('A', INF, 16, 0.0),
        ('A', 0.1, 16, 2.0, 4), ('A', 0.1, 16, 2.0, 8), ('A', 0.25, 16, 2.0, 16), ('A', INF, 16, 0.0, 8),
        ('A', 0.25, 16, 2.0, 16, 'mc'), ('A', 0.1, 16, 2.0, 4, 'mc'), ('A', 0.25, 16, 2.0, None, 'blc'),
        ('A', 0.25, 16, 2.0, 16, 'blc')],
    2: [('B', 0.1, 16, 8.0), ('B', 0.1, 16, 1.0), ('B', 0.1, 64, 8.0), ('B', 0.1, 64, 2.0),
        ('B', 0.25, 64, 8.0), ('B', 0.25, 16, 2.0), ('B', 1.0, 64, 0.0), ('B', 1.0, 16, 4.0),
        ('B', INF, 16, 0.0),
        ('B', 0.1, 16, 8.0, None, 'mc'), ('B', 0.1, 64, 2.0, None, 'mc'), ('B', 0.1, 16, 8.0, None, 'blc'),
        ('B', 0.1, 64, 2.0, None, 'blc')],
    3: [('C', 0.1, 64, 4.0), ('C', 0.1, 64, 2.0), ('C', 0.1, 16, 1.0), ('C', 0.25, 64, 2.0),
        ('C', 0.25, 16, 8.0), ('C', 1.0, 64, 0.0), ('C', 1.0, 16, 1.0), ('B', 0.1, 16, 8.0),
        ('B', 0.25, 16, 2.0), ('C', INF, 16, 0.0),
        ('C', 0.1, 64, 4.0, None, 'mc'), ('C', 0.25, 16, 8.0, None, 'mc'), ('C', 0.1, 64, 4.0, None, 'blc'),
        ('B', 0.1, 16, 8.0, None, 'mc')],
}
# 粗化不应吃掉并行度：逐级“分叉-汇合”的长链（如 case_016）按 in-tree 合并后，每级的并行分支并成一个簇，
# 簇图几乎是一条链。簇图并行度低于 GRAIN_THETA * min(N, 算子级并行度) 时，该候选另加粒度 ×4、×16…
# 直到 MAX_GRAIN 的版本，由模型挑选。100 个用例中只有 case_016、case_024、case_051 触发（均不在 16 个代表用例中）。
GRAIN_THETA = 0.75
MAX_GRAIN = 4096


def candidate_params(G, n_cores, problem, hw):
    target = GRAIN_THETA * min(n_cores, G.parallelism)
    collapsed = {}
    out = []
    for params in CANDIDATES[problem]:
        if params not in out:
            out.append(params)
        scene, split, grain = params[:3]
        if split == INF:
            continue
        mc = 'mc' in flags_of(params)
        if (split, grain, mc) not in collapsed:
            clusters, _ = build_clusters(G, n_cores, split, grain, mem_cap(hw) if mc else None)
            collapsed[(split, grain, mc)] = cluster_parallelism(G, clusters) < target
        g = grain * 4
        while collapsed[(split, grain, mc)] and g <= MAX_GRAIN:
            finer = (scene, split, g) + params[3:]
            if finer not in out:
                out.append(finer)
            g *= 4
    return out


def single_subgraph_plan(G, n_cores):
    return {'node_to_subgraph': {str(u): 0 for u in G.compute},
            'core_schedules': [[0]] + [[] for _ in range(n_cores - 1)]}


def flags_of(params):
    return set(params[5].split('+')) if len(params) > 5 else set()


def mem_cap(hw):
    return {'L1': MEM_FRAC * hw.l1, 'UB': MEM_FRAC * hw.ub}


def build_candidate(G, n_cores, hw, scene, split_ratio, grain, locality, task_div=None, opts=''):
    """opts：'+' 连接的策略开关。mc 内存感知聚簇，mr 调度时维护驻留集合，blc / child 见 sched.py。"""
    flags = set(opts.split('+')) if opts else set()
    clusters, _ = build_clusters(G, n_cores, split_ratio, grain, mem_cap(hw) if 'mc' in flags else None)
    task_cap = G.total_cycles / (n_cores * task_div) if task_div else None
    S = schedule(G, clusters, n_cores, scene, hw, locality, task_cap, flags, mem_cap(hw) if 'mr' in flags else None)
    if scene == 'A':
        plan = plan_scene_a(clusters, S)
    else:
        plan = plan_scene_b(clusters, S, G.total_cycles / (n_cores * 40.0))
    n_sub = validate_plan(G, plan)
    pred, spill = predict(G, plan, S, scene, hw)
    return plan, pred, n_sub, len(clusters), spill


def predict(G, plan, S, scene, hw):
    """返回 (调度模型 makespan + spill 时间, spill 字节)。"""
    spill, _, feasible = estimate_spill(G, plan, scene, hw)
    if not feasible:
        return INF, spill
    util = min(1.0, (S.ddr_bytes + spill) / (hw.bandwidth * max(S.makespan, 1.0)))
    return S.makespan + SPILL_ALPHA * util * spill / hw.bandwidth, spill


def single_candidate(G, n_cores, hw, scene):
    """单子图保底方案，同样用 调度模型 + spill 预测。"""
    plan = single_subgraph_plan(G, n_cores)
    pred, _ = predict(G, plan, schedule(G, single_cluster(G), n_cores, scene, hw), scene, hw)
    return pred, 'single', plan, 1


def plan_signature(plan):
    """不同参数常生成完全相同的方案，复核前按内容去重。"""
    return hash((tuple(sorted(plan['node_to_subgraph'].items())),
                 tuple(tuple(order) for order in plan['core_schedules'])))


def pad_plan(plan, n_cores):
    """少核方案补空核后仍是合法的 n_cores 核方案（用于保证核数增加时不变差）。"""
    cs = [list(x) for x in plan['core_schedules']]
    return {'node_to_subgraph': plan['node_to_subgraph'], 'core_schedules': cs + [[]] * (n_cores - len(cs))}


def solve(G, n_cores, problem, hw, verify=0, case=None, log=None, extra=()):
    """生成候选并选优。extra 为 (名称, 方案, 模型预测或 None)，如上一核数的最优方案。

    verify=0 时：取候选、单子图保底方案与 extra 中模型预测（含 spill）最小者。
    verify>0 时：每个调度场景取模型排名前 verify 个不同的方案，再加上朴素基线、单子图保底方案和
    extra，一起交给官方评估器复核，返回官方 makespan 最小者。
    """
    # 1 核时同样生成候选：切成多个子图会改变核内执行顺序，可能减少官方单核基线中的 spill。
    # 预测值已计入 spill，同一调度场景的候选统一排序；问题3 中场景 B 的候选按无 Cache 预测，单独成组。
    by_scene = {}
    for params in candidate_params(G, n_cores, problem, hw):
        t0 = time.time()
        plan, pred, n_sub, n_cl, spill = build_candidate(G, n_cores, hw, *params)
        by_scene.setdefault(params[0], []).append((pred, params, plan, n_sub))
        if log:
            log('  候选 %s: 簇=%d 子图=%d spill=%dB 预测=%.0f (%.1fs)' % (
                params, n_cl, n_sub, spill, pred, time.time() - t0))
    main_scene = {1: 'A', 2: 'B', 3: 'C'}[problem]
    for lst in by_scene.values():
        lst.sort(key=lambda c: c[0])
    best = by_scene[main_scene][0]
    extra = [(pr, name, pad_plan(p, n_cores), len(set(p['node_to_subgraph'].values())))
             for name, p, pr in extra]
    # 单子图方案在场景 A 评估器下最慢（开销随 Task 内算子数近似平方增长）；
    # 已有少核方案兜底、或模型明显优于单核时跳过。
    if n_cores == 1 or (not extra and best[0] > 0.85 * G.total_cycles):
        extra.insert(0, single_candidate(G, n_cores, hw, main_scene))
    official = None
    if not (verify and case):
        for c in extra:
            if c[0] is not None and c[0] < best[0]:
                best = c
    else:
        from .evaluate import run_official
        pool, seen = [], set()

        def add(c):
            sig = plan_signature(c[2])
            if sig in seen:
                return False
            seen.add(sig)
            pool.append(c)
            return True

        for lst in by_scene.values():
            k = 0
            for c in lst:
                if k < verify:
                    k += add(c)
            for c in lst:
                if c[1][1] == INF:
                    add(c)
        for c in extra:
            add(c)
        with tempfile.TemporaryDirectory() as tmp:
            for pred, params, plan, n_sub in pool:
                path = os.path.join(tmp, 'plan.json')
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(plan, f)
                ms = run_official(case, problem, path, tmp, tag='verify')['makespan']
                if log:
                    log('  复核 %s: 官方 makespan=%d' % (params, ms))
                if official is None or ms < official:
                    official, best = ms, (pred, params, plan, n_sub)
    pred, params, plan, n_sub = best
    return plan, {'predicted': pred, 'params': params, 'subgraphs': n_sub, 'official': official}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('case', help='用例名，如 case_019')
    ap.add_argument('--cores', type=int, default=4)
    ap.add_argument('--problem', type=int, choices=(1, 2, 3), default=1)
    ap.add_argument('--verify', type=int, default=0, help='用官方评估器复核模型排名前 K 的候选')
    ap.add_argument('-o', '--output', required=True)
    args = ap.parse_args(argv)
    hw = load_config()
    t0 = time.time()
    G = Graph(case_path(args.case))
    plan, info = solve(G, args.cores, args.problem, hw, args.verify, args.case, log=print)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(plan, f)
    print('%s P%d N=%d: 子图=%d 预测=%s 官方=%s 参数=%s 用时 %.1fs -> %s' % (
        G.name, args.problem, args.cores, info['subgraphs'], info['predicted'], info['official'],
        info['params'], time.time() - t0, args.output))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
