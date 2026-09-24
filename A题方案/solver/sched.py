"""通信感知列表调度（HEFT/ETF 变体）+ 算子级顺序发射核模型。

scene='A'：问题 1，子图即 Task，Task 级屏障（跨核 +1000、同核 +100），调度时显式维护 Task；
scene='B'：问题 2，同核子图合并，算子级跨核依赖（COPY_OUT + 500 + COPY_IN）；
scene='C'：问题 3，在 B 的基础上加入 FIFO 只读 L2 Cache 的命中模型。

核模型：PIPE_M / PIPE_V 各自按序发射（与官方 PIPE_SLOTS=1 一致），MTE2 按序搬入，
共享 DDR 带宽用时间桶近似；簇内算子按官方 step1 同规则的 DFS 序执行。

可选策略 opts：
  'blc'   优先级取向上秩 + 最大入边通信（Kulagina 等 CCGrid 2025 的 BLC），先执行能释放大输入的簇；
  'child' 关键子任务前瞻（Topcuoglu 等 TPDS 2002 第 6 节 B1）：簇与通信量最大、且其余前驱都已调度的
          后继放到使该后继最早完成的同一核；
  mem_cap 场景 B/C 按 L1/UB 容量维护每核已搬入张量的 LRU 驻留集合（HEFTM 的试分配内存检查），
          被挤出的张量再次使用时重新搬入并计入流量。
"""
import heapq
from collections import OrderedDict

from .graph import cluster_topo


class DDRBins:
    """共享 DDR 带宽的时间桶近似：每桶容量 bw*width 字节，搬运依次占用剩余容量。"""

    def __init__(self, bw, width=500.0):
        self.bw = bw
        self.width = width
        self.cap = bw * width
        self.used = {}

    def transfer(self, start, nbytes, commit):
        if nbytes <= 0:
            return start
        bw, width, cap, used = self.bw, self.width, self.cap, self.used
        i = int(start // width)
        remaining = float(nbytes)
        first = True
        writes = []
        while True:
            u = used.get(i, 0.0)
            room = cap - u
            if first:
                room = min(room, bw * ((i + 1) * width - start))
                first = False
            if room > 1e-9:
                take = min(room, remaining)
                writes.append((i, take))
                remaining -= take
                if remaining <= 1e-9:
                    end = max(start + nbytes / bw, i * width + (u + take) / bw)
                    break
            i += 1
        if commit:
            for j, take in writes:
                used[j] = used.get(j, 0.0) + take
        return end


class CacheModel:
    """FIFO 只读 Cache：只有 COPY_IN 完成时插入，命中不刷新顺序，超过容量的张量不缓存。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.entries = {}
        self.cum = 0

    def present(self, tid, t):
        e = self.entries.get(tid)
        return e is not None and e[0] <= t and self.cum - e[1] <= self.capacity

    def insert(self, tid, size, t):
        if size > self.capacity:
            return
        e = self.entries.get(tid)
        if e is not None and self.cum - e[1] <= self.capacity:
            return
        self.entries[tid] = (t, self.cum)
        self.cum += size


class Schedule:
    def __init__(self, n_cores, n_clusters):
        self.n_cores = n_cores
        self.core = [-1] * n_clusters
        self.start = [0.0] * n_clusters
        self.fin = [0.0] * n_clusters
        self.seq = []            # 调度（提交）顺序
        self.task_of = [-1] * n_clusters
        self.tasks = []          # 场景 A：[{'core', 'clusters', 'release', 'fin'}]
        self.makespan = 0.0
        self.core_end = [0.0] * n_cores
        self.traffic = 0.0       # 模型估计的新增搬运字节
        self.ddr_bytes = 0.0     # 模型中经过共享 DDR 的全部字节


class ListScheduler:
    def __init__(self, G, clusters, n_cores, scene, hw, locality=0.0, task_cap=None, opts=(), mem_cap=None):
        self.G = G
        self.cl = clusters
        self.N = n_cores
        self.scene = scene
        self.hw = hw
        self.bw = hw.bandwidth
        self.locality = locality
        self.task_cap = task_cap   # 场景 A：单个 Task 的计算量上限，超过即在同核新开 Task（限制工作集、避免 spill）
        self.delay = hw.cross_wait_a if scene == 'A' else hw.cross_delay_b
        self.hit_cost = hw.bandwidth / hw.cache_bandwidth
        self.blc = 'blc' in opts
        self.child = 'child' in opts
        self.mem_cap = mem_cap if scene != 'A' else None

    # ---------------- 优先级：带平均通信代价的向上秩 ----------------
    def upward_rank(self):
        p = (self.N - 1) / self.N if self.N > 1 else 0.0
        rank = [0.0] * len(self.cl)
        for cid in reversed(cluster_topo(self.cl)):
            c = self.cl[cid]
            best = 0.0
            for s in c.succs:
                nbytes = sum(self.G.tensor_size[t] for t in self.cl[s].preds[cid])
                best = max(best, rank[s] + p * (self.delay + 2 * nbytes / self.bw))
            rank[cid] = c.solo + best
        return rank

    def priority(self):
        rank = self.upward_rank()
        if not self.blc:
            return rank
        p = (self.N - 1) / self.N if self.N > 1 else 0.0
        size = self.G.tensor_size
        for c in self.cl:
            inc = max((p * (self.delay + 2 * sum(size[t] for t in tids) / self.bw) for tids in c.preds.values()),
                      default=0.0)
            inc = max([inc] + [s / self.bw for s in c.inputs.values()])
            rank[c.cid] += inc
        return rank

    def _critical_child(self, c, indeg):
        """B1：通信量最大的后继，且 c 是它最后一个未调度的前驱。"""
        size = self.G.tensor_size
        best, best_bytes = None, 0
        for s in c.succs:
            if indeg[s] == 1:
                b = sum(size[t] for t in self.cl[s].preds[c.cid])
                if b > best_bytes:
                    best, best_bytes = s, b
        return best

    def _child_finish(self, c, d, k, f):
        """c 在核 k 上于 f 完成时，其关键子任务 d 同放核 k 的估计完成时间。"""
        S, size = self.S, self.G.tensor_size
        ready = f
        for p, tids in self.cl[d].preds.items():
            if p == c.cid:
                continue
            if S.core[p] == k:
                a = S.fin[p]
            else:
                a = S.fin[p] + self.delay + 2 * sum(size[t] for t in tids) / self.bw
            if a > ready:
                ready = a
        return ready + self.cl[d].solo

    # ---------------- 主循环 ----------------
    def run(self):
        cl, N = self.cl, self.N
        S = Schedule(N, len(cl))
        self.S = S
        self.tM = [0.0] * N
        self.tV = [0.0] * N
        self.t2 = [0.0] * N
        self.end = [0.0] * N
        self.ddr = DDRBins(self.bw)
        self.cache = CacheModel(self.hw.cache_capacity) if self.scene == 'C' else None
        self.loaded = [dict() for _ in range(N)]
        self.lru = [OrderedDict() for _ in range(N)]
        self.resid = [dict.fromkeys(('L1', 'UB'), 0.0) for _ in range(N)]
        self.read_once = set()
        self.cur = [None] * N

        rank = self.priority()
        indeg = [len(c.preds) for c in cl]
        pinned = {}
        heap = [(-rank[c.cid], c.cid) for c in cl if indeg[c.cid] == 0]
        heapq.heapify(heap)
        while heap:
            _, cid = heapq.heappop(heap)
            c = cl[cid]
            cores = (pinned.pop(cid),) if cid in pinned else range(N)
            options = []
            for k in cores:
                for option in self._options(k):
                    res = self._evaluate(c, k, option, commit=False)
                    if res is None:
                        continue
                    f, traffic = res
                    key = (f + self.locality * traffic / self.bw, f, self.tM[k] + self.tV[k], k)
                    options.append((key, k, option, f))
            best = min(options)
            if self.child and len(cores) > 1:
                d = self._critical_child(c, indeg)
                if d is not None:
                    best = min(options, key=lambda o: (self._child_finish(c, d, o[1], o[3]), o[0]))
                    pinned[d] = best[1]
            _, k, option, _ = best
            self._evaluate(c, k, option, commit=True)
            S.seq.append(cid)
            for s in c.succs:
                indeg[s] -= 1
                if indeg[s] == 0:
                    heapq.heappush(heap, (-rank[s], s))
        if self.scene == 'A':
            S.core_end = [0.0] * N
            for t in S.tasks:
                S.core_end[t['core']] = max(S.core_end[t['core']], t['fin'])
        else:
            S.core_end = list(self.end)
        S.makespan = max(S.core_end)
        S.ddr_bytes = sum(self.ddr.used.values())
        return S

    def _options(self, k):
        if self.scene != 'A':
            return ('b',)
        cur = self.cur[k]
        if cur is None or cur['closed'] or (self.task_cap and cur['work'] >= self.task_cap):
            return ('new',)
        return ('join', 'new')

    # ---------------- 搬运 ----------------
    def _load(self, tid, size, start, commit):
        if self.cache is not None and self.cache.present(tid, start):
            return start + size / self.hw.cache_bandwidth, True
        end = self.ddr.transfer(start, size, commit)
        if commit and self.cache is not None:
            self.cache.insert(tid, size, end)
        return end, False

    def _task_ready(self, p, tid):
        """场景 A：前驱簇 p 所在 Task 完成（含该张量 COPY_OUT）的时刻。"""
        task = self.S.tasks[self.S.task_of[p]]
        return max(task['fin'], self.S.fin[p] + self.G.tensor_size[tid] / self.bw)

    # ---------------- 单个候选（核, 选项）的评估 / 提交 ----------------
    def _evaluate(self, c, k, option, commit):
        S, scene = self.S, self.scene
        core, fin_c, task_of = S.core, S.fin, S.task_of
        traffic = 0.0
        if scene == 'A':
            cur = self.cur[k]
            if option == 'join':
                release = cur['release']
                for p, tids in c.preds.items():
                    if core[p] != k:
                        for tid in tids:
                            if self._task_ready(p, tid) + self.hw.cross_wait_a > release:
                                return None
                tM, tV, t2 = self.tM[k], self.tV[k], self.t2[k]
                loaded = cur['loaded']
                task_id = cur['id']
            else:
                release = cur['fin'] + self.hw.same_wait_a if cur is not None else 0.0
                for p, tids in c.preds.items():
                    if core[p] != k:
                        for tid in tids:
                            release = max(release, self._task_ready(p, tid) + self.hw.cross_wait_a)
                tM = tV = t2 = release
                loaded = {}
                task_id = None
            base = release
        else:
            tM, tV, t2 = self.tM[k], self.tV[k], self.t2[k]
            loaded = self.loaded[k]
            task_id = None
            base = 0.0

        new = {}
        used = {}
        f = []
        first = None
        for is_m, cyc, intra, ext in c.plan:
            r = base
            for j in intra:
                if f[j] > r:
                    r = f[j]
            for tid, size, q in ext:
                if q >= 0 and core[q] == k and (scene != 'A' or task_of[q] == task_id):
                    a = fin_c[q]
                else:
                    used[tid] = size
                    a = loaded.get(tid)
                    if a is None:
                        a = new.get(tid)
                    if a is None:
                        if q < 0:
                            a, hit = self._load(tid, size, t2, commit)
                            if tid in self.read_once or scene == 'A':
                                traffic += size * self.hit_cost if hit else size
                        elif scene == 'A':
                            a, _ = self._load(tid, size, t2, commit)
                            traffic += 2 * size
                        else:
                            out_end = self.ddr.transfer(fin_c[q], size, commit)
                            a, hit = self._load(tid, size, max(out_end + self.delay, t2), commit)
                            traffic += size + (size * self.hit_cost if hit else size)
                        t2 = a
                        new[tid] = a
                if a > r:
                    r = a
            if is_m:
                s = r if r > tM else tM
                tM = s + cyc
                f.append(tM)
            else:
                s = r if r > tV else tV
                tV = s + cyc
                f.append(tV)
            if first is None or s < first:
                first = s
        fin = max(f)
        if not commit:
            return fin, traffic

        # ---------------- 提交 ----------------
        cid = c.cid
        core[cid] = k
        S.start[cid] = first
        fin_c[cid] = fin
        S.traffic += traffic
        loaded.update(new)
        if self.mem_cap is not None:
            self._retain(k, used)
        self.read_once.update(c.inputs)
        out_end = self.ddr.transfer(fin, c.out_bytes, True) if c.out_bytes else fin
        self.tM[k], self.tV[k], self.t2[k] = tM, tV, t2
        self.end[k] = max(self.end[k], out_end)
        if scene == 'A':
            if option == 'new':
                if self.cur[k] is not None:
                    self.cur[k]['closed'] = True
                task = {'id': len(S.tasks), 'core': k, 'clusters': [], 'release': release,
                        'fin': release, 'closed': False, 'loaded': loaded, 'work': 0.0}
                S.tasks.append(task)
                self.cur[k] = task
            task = self.cur[k]
            task['clusters'].append(cid)
            task['work'] += c.M + c.V
            task['fin'] = max(task['fin'], out_end)
            task_of[cid] = task['id']
            for p in c.preds:
                if core[p] != k:
                    S.tasks[task_of[p]]['closed'] = True
        return fin, traffic

    def _retain(self, k, used):
        """刚用过的张量移到 LRU 末尾；超出容量时从最久未用者挤出，挤出后再用须重新搬入。"""
        lru, resid, cap, pos = self.lru[k], self.resid[k], self.mem_cap, self.G.tensor_pos
        for tid, size in used.items():
            if tid in lru:
                lru.move_to_end(tid)
            else:
                lru[tid] = size
                resid[pos[tid]] += size
        for T in ('L1', 'UB'):
            if resid[T] <= cap[T]:
                continue
            for tid in list(lru):
                if resid[T] <= cap[T]:
                    break
                if pos[tid] != T or tid in used:
                    continue
                resid[T] -= lru.pop(tid)
                self.loaded[k].pop(tid, None)


def schedule(G, clusters, n_cores, scene, hw, locality=0.0, task_cap=None, opts=(), mem_cap=None):
    return ListScheduler(G, clusters, n_cores, scene, hw, locality, task_cap, opts, mem_cap).run()
