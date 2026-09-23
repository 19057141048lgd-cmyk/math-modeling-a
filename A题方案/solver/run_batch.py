"""批量实验：多用例 × 1~5 核 × 问题 1/2/3，官方评估 + 汇总表 + 加速比曲线。

用法（在 A题方案 目录下）：
    python -m solver.run_batch --cases case_019 case_050 --cores 1 2 3 4 5 --problems 1 2 3 --verify 3
    python -m solver.run_batch --cases all --workers 8 --out results_all   # 全部 100 个用例（耗时数小时）
结果写入 --out 目录（默认 results/）：plans/ 方案文件，eval/ 官方输出，summary.csv，table_p*.md，*.png。
官方单核基线与方案无关，统一缓存在 results/single/。
每个用例完成后写入 rows/<用例>.json；中断后用相同命令重跑，只会计算未完成的用例。
summary.csv 中 pick_makespan 是 solve() 选中方案的官方 makespan；只用模型时若兜底方案更快，
makespan 取兜底方案的结果（见 fallback）。
"""
import argparse
import csv
import glob
import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import SOLUTION_DIR, case_path, data_dir, load_config
from .evaluate import run_official, single_core_result
from .graph import Graph
from .solve import pad_plan, plan_signature, solve

RESULTS = os.path.join(SOLUTION_DIR, 'results')


def fallback(case, q, n, cur, best, prev, out):
    """只用模型时，选中方案的官方结果可能比兜底方案差。上一核数的方案补空核后官方 makespan 不变，
    直接比较；问题3 若仍慢于同核数问题2 的结果，再把问题2 的方案交问题3 评估器评估一次。
    cur 与 best 的值均为 (方案, 模型预测, 子图数, 参数, 官方结果)。"""
    if prev:
        p = best[(q, max(prev))]
        if p[4]['makespan'] < cur[4]['makespan']:
            cur = (pad_plan(p[0], n), p[1], p[2], 'prev_n%d' % max(prev), p[4])
    p = best.get((2, n))
    if q == 3 and p and p[4]['makespan'] < cur[4]['makespan'] and plan_signature(p[0]) != plan_signature(cur[0]):
        r = run_official(case, 3, os.path.join(out, 'plans', '%s_p2_n%d.json' % (case, n)),
                         os.path.join(out, 'eval'), tag='p3_n%d_p2plan' % n)
        if r['makespan'] < cur[4]['makespan']:
            cur = (p[0], p[1], p[2], 'p2_plan', r)
    return cur


def case_job(case, cores, problems, verify, out):
    """同一用例内按 问题 -> 核数 顺序求解：上一核数的最优方案（补空核）与同核数问题 2 的方案
    作为额外候选参与复核，保证加速比随核数不下降、只读 Cache 不劣于无 L2。"""
    key = {'cores': sorted(cores), 'problems': sorted(problems), 'verify': verify}
    done = os.path.join(out, 'rows', case + '.json')
    if os.path.exists(done):
        with open(done, encoding='utf-8') as f:
            saved = json.load(f)
        if saved['key'] == key:
            return saved['rows']
    base = single_core_result(case, os.path.join(RESULTS, 'single'))
    single = base['makespan']
    G = Graph(case_path(case))
    hw = load_config()
    best = {}
    rows = []
    for q in sorted(problems):
        for n in sorted(cores):
            t0 = time.time()
            extra = []
            prev = [m for m in cores if m < n and (q, m) in best]
            if prev:
                extra.append(('prev_n%d' % max(prev),) + best[(q, max(prev))][:2])
            if q == 3 and (2, n) in best:
                extra.append(('p2_plan',) + best[(2, n)][:2])
            plan, info = solve(G, n, q, hw, verify=verify, case=case, extra=extra)
            solve_s = time.time() - t0
            path = os.path.join(out, 'plans', '%s_p%d_n%d.json' % (case, q, n))
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(plan, f)
            t1 = time.time()
            if q == 1 and n == 1 and info['subgraphs'] == 1:
                # 1 核单子图方案的问题1 评估结果与官方单核基线相同，大图上这是最慢的一次评估
                r = dict(base, eval_seconds=0.0)
            else:
                r = run_official(case, q, path, os.path.join(out, 'eval'), tag='p%d_n%d' % (q, n))
            cur = (plan, info['predicted'], info['subgraphs'], info['params'], r)
            if not verify:
                cur = fallback(case, q, n, cur, best, prev, out)
            best[(q, n)] = cur
            plan, pred, n_sub, params, res = cur
            if res is not r:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(plan, f)
            rows.append({'case': case, 'problem': q, 'cores': n, 'single_makespan': single,
                         'makespan': res['makespan'], 'pick_makespan': r['makespan'],
                         'speedup': round(single / res['makespan'], 4),
                         'added_copy_bytes': res['added_copy_bytes'],
                         'spill_added_copy_bytes': res['spill_added_copy_bytes'],
                         'cache_hit_rate': res['cache_hit_rate'], 'subgraphs': n_sub,
                         'predicted': pred, 'params': str(params),
                         'solve_seconds': round(solve_s, 1), 'eval_seconds': round(time.time() - t1, 1)})
    with open(done, 'w', encoding='utf-8') as f:
        json.dump({'key': key, 'rows': rows}, f)
    return rows


def write_tables(rows, singles, out):
    by = {(r['case'], r['problem'], r['cores']): r for r in rows}
    cases = sorted({r['case'] for r in rows})
    cores = sorted({r['cores'] for r in rows})
    for q in sorted({r['problem'] for r in rows}):
        lines = ['| 用例 | 单核 makespan | ' + ' | '.join('%d核 makespan / 加速比' % n for n in cores) + ' |',
                 '|' + '---|' * (2 + len(cores))]
        for case in cases:
            cells = []
            for n in cores:
                r = by.get((case, q, n))
                cells.append('%d / %.2f' % (r['makespan'], singles[case] / r['makespan']) if r else '-')
            lines.append('| %s | %d | %s |' % (case, singles[case], ' | '.join(cells)))
        avg = []
        for n in cores:
            sp = [singles[c] / by[(c, q, n)]['makespan'] for c in cases if (c, q, n) in by]
            avg.append('%.3f' % (sum(sp) / len(sp)) if sp else '-')
        lines.append('| **平均加速比** | | %s |' % ' | '.join(avg))
        if q == 3:
            lines += ['', '只读 Cache 相对无 L2（问题 2 方案、问题 2 评估）的加速比与命中率：', '',
                      '| 用例 | ' + ' | '.join('%d核 加速比 / 命中率' % n for n in cores) + ' |',
                      '|' + '---|' * (1 + len(cores))]
            for case in cases:
                cells = []
                for n in cores:
                    r3, r2 = by.get((case, 3, n)), by.get((case, 2, n))
                    if r3 and r2:
                        cells.append('%.3f / %.1f%%' % (r2['makespan'] / r3['makespan'],
                                                        100 * (r3['cache_hit_rate'] or 0)))
                    else:
                        cells.append('-')
                lines.append('| %s | %s |' % (case, ' | '.join(cells)))
        with open(os.path.join(out, 'table_p%d.md' % q), 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


def plot(rows, singles, out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('未安装 matplotlib，跳过绘图')
        return
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    avg = defaultdict(list)
    for r in rows:
        avg[(r['problem'], r['cores'])].append(singles[r['case']] / r['makespan'])
    cores = sorted({r['cores'] for r in rows})
    fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
    names = {1: '问题1 场景A（Task 屏障）', 2: '问题2 场景B（无 L2）', 3: '问题3 场景B + 只读 Cache'}
    for q in sorted({r['problem'] for r in rows}):
        ys = [sum(avg[(q, n)]) / len(avg[(q, n)]) for n in cores if avg[(q, n)]]
        ax.plot(cores[:len(ys)], ys, marker='o', label=names[q])
    base_csv = next((p for p in (os.path.join(out, 'baseline.csv'), os.path.join(RESULTS, 'baseline.csv'))
                     if os.path.exists(p)), None)
    if base_csv:
        cases = {r['case'] for r in rows}
        base, covered = defaultdict(list), set()
        with open(base_csv, encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):
                if r['case'] in cases:
                    covered.add(r['case'])
                    base[int(r['cores'])].append(float(r['p1_speedup']))
        if covered == cases:  # 基线只覆盖部分用例时，其平均值与上面的曲线不可比
            xs = [1] + sorted(base)
            ax.plot(xs, [1.0] + [sum(base[n]) / len(base[n]) for n in xs[1:]], marker='s', linestyle=':',
                    color='black', label='朴素基线（独立作业 LPT）')
    ax.plot(cores, cores, '--', color='gray', label='线性加速')
    ax.set_xlabel('核数')
    ax.set_ylabel('平均加速比（相对官方单核）')
    ax.set_xticks(cores)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'speedup_curves.png'))
    if any(r['problem'] == 3 for r in rows) and any(r['problem'] == 2 for r in rows):
        by = {(r['case'], r['problem'], r['cores']): r for r in rows}
        cases = sorted({r['case'] for r in rows})
        fig, ax = plt.subplots(figsize=(6, 4.2), dpi=150)
        for q, label in ((2, '无 L2（问题2）'), (3, '只读 Cache（问题3）')):
            ys = []
            for n in cores:
                sp = [singles[c] / by[(c, q, n)]['makespan'] for c in cases if (c, q, n) in by]
                ys.append(sum(sp) / len(sp))
            ax.plot(cores, ys, marker='o', label=label)
        ax.set_xlabel('核数')
        ax.set_ylabel('平均加速比（相对官方单核）')
        ax.set_xticks(cores)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'cache_compare.png'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', nargs='+', required=True, help='用例名列表，或 all')
    ap.add_argument('--cores', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--problems', type=int, nargs='+', default=[1, 2, 3])
    ap.add_argument('--verify', type=int, default=3)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--out', default=RESULTS, help='结果目录（默认 results/）')
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    cases = args.cases
    if cases == ['all']:
        cases = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(data_dir(), 'case_*.json')))
    # 总耗时取决于最慢的大图，按文件大小从大到小提交
    cases = sorted(cases, key=lambda c: -os.path.getsize(case_path(c)))
    for sub in ('plans', 'eval', 'rows'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, 'single'), exist_ok=True)

    t0 = time.time()
    rows, failed = [], []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(case_job, c, args.cores, args.problems, args.verify, out): c for c in cases}
        for fut in as_completed(futs):
            try:
                case_rows = fut.result()
            except Exception as e:
                failed.append(futs[fut])
                print('%s 失败：%s' % (futs[fut], e), flush=True)
                continue
            for r in case_rows:
                rows.append(r)
                picked = r.get('pick_makespan', r['makespan'])
                print('%s P%d N=%d makespan=%d 加速比=%.2f 子图=%d 参数=%s (求解 %.0fs, 评估 %.0fs)%s' % (
                    r['case'], r['problem'], r['cores'], r['makespan'], r['speedup'],
                    r['subgraphs'], r['params'], r['solve_seconds'], r['eval_seconds'],
                    ' 兜底（选中方案 %d）' % picked if picked > r['makespan'] else ''), flush=True)
    if failed:
        print('失败 %d 个用例：%s（用相同命令重跑只会重算未完成的用例）' % (len(failed), ' '.join(sorted(failed))))
    if not rows:
        return

    rows.sort(key=lambda r: (r['case'], r['problem'], r['cores']))
    with open(os.path.join(out, 'summary.csv'), 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['case', 'problem', 'cores', 'single_makespan', 'makespan', 'pick_makespan',
                                          'speedup',
                                          'added_copy_bytes', 'spill_added_copy_bytes', 'cache_hit_rate',
                                          'subgraphs', 'predicted', 'params', 'solve_seconds', 'eval_seconds'])
        w.writeheader()
        w.writerows(rows)
    singles = {r['case']: r['single_makespan'] for r in rows}
    write_tables(rows, singles, out)
    plot(rows, singles, out)
    print('全部完成，用时 %.0fs，结果见 %s' % (time.time() - t0, out))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
