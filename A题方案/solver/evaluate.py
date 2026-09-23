"""调用官方评估程序（子进程），返回 makespan 与次要指标。"""
import json
import os
import subprocess
import sys
import time

from .config import case_path, code_dir, data_dir

SCRIPTS = {
    0: 'singlecore_evaluate.py',
    1: 'multicore_cut_evaluate_problem_1.py',
    2: 'multicore_cut_evaluate_problem_2.py',
    3: 'multicore_cut_evaluate_problem_3.py',
}


def run_official(case, problem, plan_path=None, out_dir='.', tag=None, keep_trace=False, timeout=None):
    """problem=0 为单核基线；1/2/3 为对应问题的多核评估。超过 timeout 秒抛出 subprocess.TimeoutExpired。"""
    os.makedirs(out_dir, exist_ok=True)
    tag = tag or ('single' if problem == 0 else 'p%d' % problem)
    res = os.path.join(out_dir, '%s_%s.json' % (case, tag))
    args = [sys.executable, os.path.join(code_dir(), SCRIPTS[problem]), case_path(case)]
    if plan_path:
        args.append(plan_path)
    trace = res[:-5] + '.trace.json' if keep_trace else os.devnull
    args += ['--config', os.path.join(data_dir(), 'config.txt'), '-o', res,
             '--trace-output', trace, '--log-output', res[:-5] + '.log']
    t0 = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=timeout)
    elapsed = time.time() - t0
    if proc.returncode != 0 or not os.path.exists(res):
        raise RuntimeError('官方评估失败 %s P%d:\n%s' % (case, problem, proc.stderr[-2000:]))
    return read_result(res, elapsed)


def read_result(res, elapsed=0.0):
    with open(res, encoding='utf-8') as f:
        out = json.load(f)
    moved = out.get('data_movement_bytes', {})
    cache = out.get('cache_stats') or {}
    return {
        'makespan': out['makespan'],
        'added_copy_bytes': moved.get('added_copy_bytes', 0),
        'spill_added_copy_bytes': moved.get('spill_added_copy_bytes', 0),
        'cache_hit_rate': cache.get('hit_rate'),
        'eval_seconds': elapsed,
    }


def single_core_result(case, out_dir):
    """单核基线带缓存：结果文件已存在时直接读取。"""
    res = os.path.join(out_dir, '%s_single.json' % case)
    if os.path.exists(res):
        return read_result(res)
    return run_official(case, 0, None, out_dir)


def single_core_makespan(case, out_dir):
    return single_core_result(case, out_dir)['makespan']
