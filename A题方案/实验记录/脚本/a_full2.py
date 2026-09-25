import csv
import os
import statistics
import sys

SOL = r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案'
sys.stdout.reconfigure(encoding='utf-8')


def load(path):
    with open(path, encoding='utf-8-sig') as f:
        return {(r['case'], int(r['problem']), int(r['cores'])): r for r in csv.DictReader(f)}


A = load(os.path.join(SOL, 'results_all', 'summary.csv'))
V = load(os.path.join(SOL, 'results', 'summary.csv'))
N5 = load(os.path.join(SOL, 'results', 'dev', 'noverify5', 'summary.csv'))
cases = sorted({k[0] for k in A})
ms = {k: float(r['makespan']) for k, r in A.items()}
single = {c: float(A[(c, 1, 1)]['single_makespan']) for c in cases}

print('== 问题2 5核加速比 < 2 的用例')
for c in cases:
    if single[c] / ms[(c, 2, 5)] < 2:
        print(' ', c, '单核', int(single[c]), '  '.join(
            'P%d: %s' % (q, ' '.join('%.2f' % (single[c] / ms[(c, q, n)]) for n in range(1, 6))) for q in (1, 2, 3)))
        for q in (1, 2):
            print('    P%d 参数' % q, [A[(c, q, n)]['params'] for n in range(1, 6)],
                  '子图', [A[(c, q, n)]['subgraphs'] for n in range(1, 6)],
                  '预测', [int(float(A[(c, q, n)]['predicted'])) for n in range(1, 6)],
                  '选中方案官方', [A[(c, q, n)]['pick_makespan'] for n in range(1, 6)])

print('\n== 问题3 劣于问题2')
for (c, q, n), r in A.items():
    if q == 3 and ms[(c, 3, n)] > ms[(c, 2, n)]:
        p2eval = os.path.exists(os.path.join(SOL, 'results_all', 'eval', '%s_p3_n%d_p2plan.json' % (c, n)))
        print('  %s N=%d P3 %d P2 %d (%.4f) 参数 %s 选中方案 %s 补评过问题2方案 %s' % (
            c, n, ms[(c, 3, n)], ms[(c, 2, n)], ms[(c, 3, n)] / ms[(c, 2, n)], r['params'], r['pick_makespan'], p2eval))

print('\n== 兜底挽回最多的组合')
fb = [(float(r['pick_makespan']) / ms[k], k, r['params']) for k, r in A.items() if float(r['pick_makespan']) > ms[k]]
for g, k, p in sorted(fb, reverse=True)[:6]:
    print('  %s P%d N=%d %.3f %s' % (k[0], k[1], k[2], g, p))
print('  按问题：' + '  '.join('P%d %d 次' % (q, sum(1 for _, k, _ in fb if k[1] == q)) for q in (1, 2, 3)))

print('\n== 16 个代表用例：全量结果 vs 16 用例测试（noverify5）vs 复核')
same = sum(abs(ms[k] - float(N5[k]['makespan'])) < 0.5 for k in N5)
print('  与 noverify5 相同 %d / %d' % (same, len(N5)))
diff = [k for k in N5 if abs(ms[k] - float(N5[k]['makespan'])) >= 0.5]
for k in diff[:8]:
    print('   不同', k, ms[k], N5[k]['makespan'], A[k]['params'], N5[k]['params'])
rat = [ms[k] / float(V[k]['makespan']) for k in V]
print('  全量结果 / 复核：平均 %.4f 最差 %.3f 相同 %d' % (statistics.mean(rat), max(rat), sum(abs(x - 1) < 1e-9 for x in rat)))

print('\n== 选中方案的预测精度（预测 / 选中方案官方 - 1）')
for q in (1, 2, 3):
    e = [float(A[(c, q, n)]['predicted']) / float(A[(c, q, n)]['pick_makespan']) - 1 for c in cases for n in range(1, 6)]
    print('  P%d：平均偏差 %+.1f%%，平均绝对误差 %.1f%%，绝对误差 10%% 以内 %d/%d，低估超过 10%% %d 个' % (
        q, 100 * statistics.mean(e), 100 * statistics.mean(abs(x) for x in e), sum(abs(x) <= 0.1 for x in e), len(e),
        sum(x < -0.1 for x in e)))

print('\n== P1 与 P2 对比（5 核）')
better1 = [c for c in cases if ms[(c, 1, 5)] < ms[(c, 2, 5)]]
print('  问题1 快于问题2 的用例 %d 个：%s' % (len(better1), better1[:12]))
print('  问题2/问题1 加速比之比：平均 %.3f' % statistics.mean(ms[(c, 1, 5)] / ms[(c, 2, 5)] for c in cases))
