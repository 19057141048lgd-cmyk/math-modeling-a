"""
schedule_step3.py — 步骤 3: 乱序流水排布 (out-of-order scheduling)

对应文档: docs/核内调度算法.md § 3

职责边界:
    - Step2 负责片上内存容量、换入换出节点和扩展图；
    - Step3 只读取扩展图，按依赖和 4 条 pipe 排布时间；
    - 所有访问 DDR 的 COPY 共享总带宽，L1↔UB COPY 使用内部带宽；
    - Step3 按重命名 tensor 的消费者引用计数执行 alloc/free 和容量门控；
    - 不分配具体地址。容量被抽象为可拆分的虚拟字节额度；
    - 额度复用时，旧 tensor 的全部读者到新 producer 之间增加同步依赖，
      用来补偿真实地址复用会产生的 WAR/WAW 假数据依赖；
    - 输出确定的逐 Pipe 顺序和补边后的执行图，供三个多核评估器直接使用。

若仍有未完成 op 且没有任何可运行/在运行 op，说明扩展图或调度器存在错误。
此时打印完整诊断并抛出 Step3SchedulingError；CLI 同时把诊断写入对应日志。
"""

import heapq
import math
from copy import deepcopy
from collections import Counter, defaultdict

from contest_io import emit_error

# 硬件模型：核内四条指令 Pipe，每条同一时刻只有 1 个在飞指令。
# 二者都是固定约束而非选手可调参数，因此写死在代码里，config.txt 不接收。
PIPES = ('PIPE_MTE2', 'PIPE_MTE3', 'PIPE_M', 'PIPE_V')
PIPE_SLOTS = 1

class Step3SchedulingError(RuntimeError):
    """扩展图无法继续发射但仍有未完成 op 时抛出。"""

def op_pipe(op):
    """Pipe 直接取节点自带的 pipe 字段，不按 op 名称猜测。"""
    return op['pipe']

def _build_graph_views(ops, edges):
    """统一解析 tensor 中转边与直接 op-op 边；DDR tensor 同样参与依赖。"""
    op_ids = {o['id'] for o in ops}
    in_tids = {o['id']: [] for o in ops}
    out_tids = {o['id']: [] for o in ops}
    direct_edges = []
    for edge in edges:
        src, dst = edge['source'], edge['target']
        src_is_op = src in op_ids
        dst_is_op = dst in op_ids
        if src_is_op and not dst_is_op:
            out_tids[src].append(dst)
        elif not src_is_op and dst_is_op:
            in_tids[dst].append(src)
        elif src_is_op and dst_is_op and src != dst:
            direct_edges.append((src, dst))

    producers = defaultdict(set)
    for op_id, tids in out_tids.items():
        for tid in tids:
            producers[tid].add(op_id)

    preds = {o['id']: set() for o in ops}
    succs = {o['id']: set() for o in ops}
    for op_id, tids in in_tids.items():
        for tid in tids:
            for producer in producers.get(tid, ()):
                if producer != op_id:
                    preds[op_id].add(producer)
                    succs[producer].add(op_id)
    for src, dst in direct_edges:
        preds[dst].add(src)
        succs[src].add(dst)
    return in_tids, out_tids, preds, succs

def _op_duration(op, in_tids, out_tids, tensor_by_id, bandwidth):
    """普通 op 使用 cycles；COPY op 优先按搬运 tensor 大小换算。"""
    op_type = op['op']
    if op_type in ('COPY_IN', 'COPY_OUT'):
        tids = out_tids[op['id']] if op_type == 'COPY_IN' else in_tids[op['id']]
        sizes = [tensor_by_id[tid]['size'] for tid in tids if tid in tensor_by_id]
        if sizes:
            return max(1, math.ceil(sum(sizes) / bandwidth))
    return max(1, op.get('cycles', 1))

def _uses_ddr_bandwidth(op, in_tids, out_tids, tensor_by_id):
    """只有端点包含 DDR 的 COPY 才进入共享 DDR 带宽池。"""
    if op['op'] not in ('COPY_IN', 'COPY_OUT'):
        return False
    tids = in_tids[op['id']] + out_tids[op['id']]
    return any(
        tid in tensor_by_id and tensor_by_id[tid].get('pos') == 'DDR'
        for tid in tids
    )

def step3_simulation(ext_graph, capacity, bandwidth, max_iter=100000):
    """按依赖、pipe、DDR 带宽和重命名 tensor 生命周期调度扩展图。

    输入约束：seq_ext是扩展图所有op的完整拓扑序，ID/端点合法，容量、带宽
    为入口校验后的合法值。
    保证来源：Step2沿Step1拓扑序插入SPILL并分配新ID，重命名后同时
    构造ext_edges和seq_ext；_build_extended_graph原样组装这份输出。
    不重查输入；仍保留本阶段新生成内存依赖/执行顺序的输出契约检查。
    capacity、bandwidth 由调用方从 config.txt 读出后传入，本函数不再内置
    默认配置。硬件模型限定每条 Pipe 同一时刻只有 PIPE_SLOTS 个在飞指令。
    """
    capacity = dict(capacity)

    ops = ext_graph['ops']
    tensors = ext_graph['tensors']
    edges = ext_graph['edges']
    seq = ext_graph['seq_ext']
    op_by_id = {o['id']: o for o in ops}
    tensor_by_id = {t['id']: t for t in tensors}

    in_tids, out_tids, preds, succs = _build_graph_views(ops, edges)
    tensor_producers = defaultdict(set)
    tensor_consumers = defaultdict(set)
    for op_id, tids in out_tids.items():
        for tid in tids:
            tensor_producers[tid].add(op_id)
    for op_id, tids in in_tids.items():
        for tid in tids:
            tensor_consumers[tid].add(op_id)

    # 每个重命名后的片上 tensor 就是一段物理驻留生命周期。输出 op 发射前
    # alloc，全部消费者完成后 free；真实 COPY_OUT 也是普通消费者。
    managed_tensors = {
        tid for tid, tensor in tensor_by_id.items()
        if tensor.get('pos') in capacity
    }
    remaining_consumers = {
        tid: len(tensor_consumers.get(tid, ())) for tid in managed_tensors
    }
    resident_tensors = set()
    memory_used = {pos: 0 for pos in capacity}
    memory_peak = {pos: 0 for pos in capacity}
    memory_events = []
    # free_credits 是“可复用容量”，只有字节数和上一个 tensor 的同步来源，
    # 没有 offset。额度允许拆分/合并，因此等价于执行前可做一次无代价内存整理，
    # 不会引入地址碎片问题。
    free_credits = {pos: [] for pos in capacity}
    memory_dependency_parts = defaultdict(lambda: {
        'bytes': 0, 'tensor_ids': set(), 'kinds': set(), 'positions': set()})

    def add_free_credit(tid):
        tensor = tensor_by_id[tid]
        consumers = sorted(tensor_consumers.get(tid, ()))
        producers = sorted(tensor_producers.get(tid, ()))
        # 有读者时必须等全部读者结束（WAR）；无读者的死输出至少要等旧写完成
        # 才能复用同一额度（WAW）。
        sources = consumers if consumers else producers
        free_credits[tensor['pos']].append({
            'bytes': tensor['size'],
            'sources': tuple(sources),
            'tid': tid,
            'kind': 'WAR' if consumers else 'WAW',
        })

    def consume_free_credit(tid, producer_op):
        tensor = tensor_by_id[tid]
        pos = tensor['pos']
        remaining = tensor['size']
        credits = free_credits[pos]
        while remaining > 0 and credits:
            credit = credits[0]
            taken = min(remaining, credit['bytes'])
            if credit['sources']:
                for source_op in credit['sources']:
                    if source_op == producer_op:
                        continue
                    part = memory_dependency_parts[(source_op, producer_op)]
                    part['bytes'] += taken
                    part['tensor_ids'].add(credit['tid'])
                    part['kinds'].add(credit['kind'])
                    part['positions'].add(pos)
            credit['bytes'] -= taken
            remaining -= taken
            if credit['bytes'] == 0:
                credits.pop(0)
        if remaining:
            message = '[STEP3 ERROR] virtual capacity credits exhausted: tid={} missing={}'.format(
                tid, remaining)
            emit_error(message)
            raise Step3SchedulingError(message)

    def allocate_tensor(tid, now, producer_op=None, kind='alloc'):
        if tid not in managed_tensors or tid in resident_tensors:
            return
        tensor = tensor_by_id[tid]
        pos, size = tensor['pos'], tensor['size']
        if producer_op is not None:
            consume_free_credit(tid, producer_op)
        memory_used[pos] += size
        resident_tensors.add(tid)
        memory_peak[pos] = max(memory_peak[pos], memory_used[pos])
        memory_events.append({
            'time': now, 'kind': kind, 'tid': tid,
            'logical_tid': tensor.get('logical_tid', tid),
            'pos': pos, 'size': size, 'op_id': producer_op,
            'used_after': memory_used[pos],
        })

    def release_tensor(tid, now, consumer_op=None):
        if tid not in resident_tensors:
            return
        tensor = tensor_by_id[tid]
        pos, size = tensor['pos'], tensor['size']
        resident_tensors.remove(tid)
        memory_used[pos] -= size
        add_free_credit(tid)
        memory_events.append({
            'time': now, 'kind': 'free', 'tid': tid,
            'logical_tid': tensor.get('logical_tid', tid),
            'pos': pos, 'size': size, 'op_id': consumer_op,
            'used_after': memory_used[pos],
        })

    # 没有片上 producer 的 tensor 视为图开始前已驻留。
    for tid in sorted(managed_tensors):
        if not tensor_producers.get(tid) and tensor_consumers.get(tid):
            allocate_tensor(tid, 0, kind='initial_alloc')
    initial_overflow = {
        pos: memory_used[pos] for pos in capacity
        if memory_used[pos] > capacity[pos]
    }
    if initial_overflow:
        message = (
            '[STEP3 ERROR] initial resident tensors exceed capacity: used={} capacity={}'
        ).format(initial_overflow, capacity)
        emit_error(message)
        raise Step3SchedulingError(message)
    for pos in capacity:
        unused = capacity[pos] - memory_used[pos]
        if unused:
            free_credits[pos].append({
                'bytes': unused, 'sources': (), 'tid': None,
                'kind': 'VIRGIN',
            })

    def allocation_requirement(op_id):
        required = {pos: 0 for pos in capacity}
        tids = []
        for tid in set(out_tids[op_id]):
            if tid not in managed_tensors or tid in resident_tensors:
                continue
            tensor = tensor_by_id[tid]
            required[tensor['pos']] += tensor['size']
            tids.append(tid)
        return required, sorted(tids)

    def can_allocate_outputs(op_id):
        required, _ = allocation_requirement(op_id)
        return all(
            memory_used[pos] + required[pos] <= capacity[pos]
            for pos in capacity
        )

    def allocate_outputs(op_id, now):
        for tid in sorted(set(out_tids[op_id])):
            allocate_tensor(tid, now, producer_op=op_id)

    def consume_inputs(op_id, now):
        for tid in set(in_tids[op_id]):
            if tid not in remaining_consumers:
                continue
            if op_id not in tensor_consumers.get(tid, ()):
                continue
            remaining_consumers[tid] -= 1
            if remaining_consumers[tid] < 0:
                message = '[STEP3 ERROR] negative tensor refcount: tid={} op={}'.format(
                    tid, op_id)
                emit_error(message)
                raise Step3SchedulingError(message)
            if remaining_consumers[tid] == 0:
                release_tensor(tid, now, consumer_op=op_id)

    def release_dead_outputs(op_id, now):
        for tid in set(out_tids[op_id]):
            if tid in remaining_consumers and remaining_consumers[tid] == 0:
                release_tensor(tid, now, consumer_op=op_id)

    seq_pos = {op_id: i for i, op_id in enumerate(seq)}
    # seq_ext 是 Step2 已验证的确定性拓扑序。Step3 在每条 Pipe 上使用它的
    # 投影作为固定发射顺序；这样多核阶段加入跨核 COPY 等待后，仍与核内
    # 调度结果保持同序，不会因局部 ready 时刻不同而重新排列 Pipe。
    planned_pipe_orders = {pipe: [] for pipe in PIPES}
    for op_id in seq:
        planned_pipe_orders[op_pipe(op_by_id[op_id])].append(op_id)
    pipe_cursor = {pipe: 0 for pipe in PIPES}
    allocation_order = [
        op_id for op_id in seq
        if any(tid in managed_tensors for tid in out_tids[op_id])
    ]
    allocation_rank = {
        op_id: rank for rank, op_id in enumerate(allocation_order)
    }
    next_allocation_rank = 0
    op_status = {op_id: 'pending' for op_id in op_by_id}
    pred_remaining = {op_id: len(preds[op_id]) for op_id in op_by_id}
    op_start, op_end = {}, {}
    issue_queues = {pipe: [] for pipe in PIPES}
    allocation_ready = set()
    executors = {pipe: [] for pipe in PIPES}
    ddr_transfer = {
        op_id: _uses_ddr_bandwidth(op, in_tids, out_tids, tensor_by_id)
        for op_id, op in op_by_id.items()
    }

    # DDR 公平共享状态。remaining_work 的单位是“独占全部 DDR 带宽时还需多少 cycle”。
    # 新的第 n 个搬运加入后，既有搬运的带宽份额从 1/(n-1) 降为 1/n，
    # 对应剩余时间按 n/(n-1) 拉长。下面按 remaining_work 排序、逐次移除
    # 最早完成项，等价实现用户指定的递推，也自然支持未来多核的 2N 条 MTE pipe。
    ddr_remaining_work = {}
    ddr_last_update = 0.0
    ddr_contention_log = []

    def advance_ddr_work(now):
        nonlocal ddr_last_update
        elapsed = now - ddr_last_update
        # projected end 会向上取整到 cycle。若某搬运在一个 cycle 内部已经
        # 完成，则余下的分数时间应立即由其他搬运共享，不能让已完成项继续占带宽。
        while elapsed > 1e-9:
            active = [
                op_id for op_id, work in ddr_remaining_work.items()
                if work > 1e-9
            ]
            if not active:
                break
            min_work = min(ddr_remaining_work[op_id] for op_id in active)
            time_to_first_finish = min_work * len(active)
            if time_to_first_finish >= elapsed - 1e-9:
                share = elapsed / len(active)
                for op_id in active:
                    ddr_remaining_work[op_id] = max(
                        0.0, ddr_remaining_work[op_id] - share)
                break
            for op_id in active:
                ddr_remaining_work[op_id] = max(
                    0.0, ddr_remaining_work[op_id] - min_work)
            elapsed -= time_to_first_finish
        ddr_last_update = now

    def reschedule_ddr_ends(now):
        """逐个移除最早项，同时计算临时 end、竞争惩罚与最终 end。"""
        if not ddr_remaining_work:
            return {}, {}, []
        ordered = sorted(
            (max(0.0, work), op_id)
            for op_id, work in ddr_remaining_work.items()
        )
        projected = {}
        provisional = {}
        stages = []
        base_cursor = float(now)
        adjusted_cursor = float(now)
        previous_work = 0.0
        active_count = len(ordered)
        i = 0
        while i < len(ordered):
            work = ordered[i][0]
            delta_work = work - previous_work
            if active_count > 1:
                # 新 op 是逐个加入的：先按加入前 n-1 路的时间轴得到临时区间 t，
                # 再用 n/(n-1) 把该区间拉长。额外惩罚为 adjusted - base。
                base_delta = delta_work * (active_count - 1)
                adjusted_delta = base_delta * active_count / (active_count - 1)
                factor = active_count / (active_count - 1)
            else:
                base_delta = delta_work
                adjusted_delta = delta_work
                factor = 1.0
            base_start = base_cursor
            adjusted_start = adjusted_cursor
            base_cursor += base_delta
            adjusted_cursor += adjusted_delta
            j = i
            while j < len(ordered) and abs(ordered[j][0] - work) <= 1e-9:
                provisional[ordered[j][1]] = base_cursor
                # 调度时间单位为 cycle；分段核算允许小数，但写回的最终 end
                # 必须向上取整，避免产生无意义的浮点时间戳。
                projected[ordered[j][1]] = int(math.ceil(adjusted_cursor - 1e-9))
                j += 1
            stages.append({
                'active_ops': sorted(op_id for _, op_id in ordered[i:]),
                'n': active_count,
                'base_start': base_start,
                'base_end': base_cursor,
                'adjusted_start': adjusted_start,
                'adjusted_before_penalty_end': adjusted_start + base_delta,
                'adjusted_end': adjusted_cursor,
                'factor': factor,
                'penalty': adjusted_delta - base_delta,
                'completed_ops': sorted(ordered[k][1] for k in range(i, j)),
            })
            active_count -= j - i
            previous_work = work
            i = j

        for op_id, end in projected.items():
            op_end[op_id] = end
        for pipe in executors:
            executors[pipe] = [
                (op_id, projected.get(op_id, end))
                for op_id, end in executors[pipe]
            ]
        return projected, provisional, stages

    def queue_if_ready(op_id):
        if op_status[op_id] == 'pending' and pred_remaining[op_id] == 0:
            pipe = op_pipe(op_by_id[op_id])
            if pipe not in issue_queues:
                message = '[STEP3 ERROR] missing pipe capacity: op={} pipe={}'.format(op_id, pipe)
                emit_error(message)
                raise Step3SchedulingError(message)
            order = planned_pipe_orders[pipe]
            cursor = pipe_cursor[pipe]
            if cursor >= len(order) or order[cursor] != op_id:
                return
            op_status[op_id] = 'ready'
            if op_id in allocation_rank:
                allocation_ready.add(op_id)
            else:
                heapq.heappush(issue_queues[pipe], (seq_pos[op_id], op_id))

    for op_id in seq:
        queue_if_ready(op_id)

    t_now = 0
    iteration = 0

    def retire_step():
        advance_ddr_work(t_now)
        retired_ddr = False
        for pipe in executors:
            still_running = []
            for op_id, end in executors[pipe]:
                if end <= t_now + 1e-9:
                    op_status[op_id] = 'done'
                    consume_inputs(op_id, t_now)
                    release_dead_outputs(op_id, t_now)
                    completed_pipe = op_pipe(op_by_id[op_id])
                    completed_cursor = pipe_cursor[completed_pipe]
                    pipe_cursor[completed_pipe] += 1
                    completed_order = planned_pipe_orders[completed_pipe]
                    if completed_cursor + 1 < len(completed_order):
                        queue_if_ready(completed_order[completed_cursor + 1])
                    if op_id in ddr_remaining_work:
                        ddr_remaining_work.pop(op_id, None)
                        retired_ddr = True
                    for succ_id in succs[op_id]:
                        pred_remaining[succ_id] -= 1
                        queue_if_ready(succ_id)
                else:
                    still_running.append((op_id, end))
            executors[pipe] = still_running
        if retired_ddr:
            reschedule_ddr_ends(t_now)

    def issue_step():
        nonlocal next_allocation_rank
        # 发射严格逐个处理。即使多个 op 的 t_now 相同，也先完成当前 op 的
        # 加入、临时 end 计算和全体 end 更新，再弹出下一个 op；不存在批量同时加入。
        issued_in_pass = True
        while issued_in_pass:
            issued_in_pass = False
            for pipe in PIPES:
                while len(executors[pipe]) < PIPE_SLOTS:
                    # 所有片上 allocation 严格沿用 Step2 的 seq_ext 顺序；执行仍可
                    # 跨 pipe 重叠。这样不会让后序分支提前占满内存并形成资源死锁。
                    op_id = None
                    if next_allocation_rank < len(allocation_order):
                        candidate = allocation_order[next_allocation_rank]
                        candidate_pipe = op_pipe(op_by_id[candidate])
                        if (candidate in allocation_ready
                                and candidate_pipe == pipe
                                and can_allocate_outputs(candidate)):
                            allocation_ready.remove(candidate)
                            op_id = candidate
                    if op_id is None and issue_queues[pipe]:
                        _, op_id = heapq.heappop(issue_queues[pipe])
                    if op_id is None:
                        break
                    issued_in_pass = True
                    rank = allocation_rank.get(op_id)
                    if rank is not None:
                        next_allocation_rank += 1
                    op_status[op_id] = 'running'
                    allocate_outputs(op_id, t_now)
                    duration = _op_duration(
                        op_by_id[op_id], in_tids, out_tids,
                        tensor_by_id, bandwidth)
                    op_start[op_id] = t_now
                    op_end[op_id] = t_now + duration
                    executors[pipe].append((op_id, op_end[op_id]))
                    if ddr_transfer[op_id]:
                        advance_ddr_work(t_now)
                        previous_count = len(ddr_remaining_work)
                        ddr_remaining_work[op_id] = float(duration)
                        active_count = len(ddr_remaining_work)
                        slowdown_factor = (
                            active_count / previous_count if previous_count else 1.0
                        )
                        projected, provisional, stages = reschedule_ddr_ends(t_now)
                        ddr_contention_log.append({
                            'time': t_now,
                            'issued_op': op_id,
                            'issue_order': len(ddr_contention_log) + 1,
                            'exclusive_end': t_now + duration,
                            'provisional_end': provisional[op_id],
                            'slowdown_factor': slowdown_factor,
                            'active_ops': sorted(ddr_remaining_work),
                            'stages': stages,
                            'projected_ends': {
                                active_id: projected[active_id]
                                for active_id in sorted(projected)
                            },
                        })

    def raise_deadlock(reason):
        status_count = Counter(op_status.values())
        queue_snapshot = {
            pipe: [op_id for _, op_id in sorted(queue)]
            for pipe, queue in issue_queues.items()
        }
        executor_snapshot = {
            pipe: sorted(running, key=lambda item: item[1])
            for pipe, running in executors.items()
        }
        lines = [
            '[STEP3 ERROR] scheduler deadlock: reason={} time={} iteration={} status={}'.format(
                reason, t_now, iteration, dict(status_count)),
            'queues={}'.format(queue_snapshot),
            'executors={}'.format(executor_snapshot),
            'ddr_remaining_work={}'.format(dict(sorted(ddr_remaining_work.items()))),
            'memory_used={} capacity={} resident_tensors={}'.format(
                memory_used, capacity, sorted(resident_tensors)),
            'allocation_front={} allocation_ready={}'.format(
                allocation_order[next_allocation_rank]
                if next_allocation_rank < len(allocation_order) else None,
                sorted(allocation_ready)),
        ]
        for op_id in seq:
            if op_status[op_id] == 'done':
                continue
            unresolved = sorted(
                pred for pred in preds[op_id] if op_status.get(pred) != 'done')
            lines.append(
                'op={} type={} pipe={} seq_pos={} status={} pred_remaining={} '
                'unresolved_preds={} in_tids={} out_tids={} alloc_required={}'.format(
                    op_id, op_by_id[op_id]['op'],
                    op_pipe(op_by_id[op_id]), seq_pos[op_id],
                    op_status[op_id], pred_remaining[op_id], unresolved,
                    in_tids[op_id], out_tids[op_id],
                    allocation_requirement(op_id)))
        message = '\n'.join(lines)
        emit_error(message)
        raise Step3SchedulingError(message)

    while True:
        iteration += 1
        if iteration > max_iter:
            raise_deadlock('max_iter_exceeded')
        retire_step()
        if all(status == 'done' for status in op_status.values()):
            break
        issue_step()
        next_times = [end for running in executors.values() for _, end in running]
        if not next_times:
            raise_deadlock('no_running_or_issuable_op')
        t_now = min(next_times)

    makespan = max(op_end.values(), default=0)
    per_pipe_timeline = {pipe: [] for pipe in PIPES}
    for op_id, start in op_start.items():
        pipe = op_pipe(op_by_id[op_id])
        per_pipe_timeline[pipe].append((op_id, start, op_end[op_id]))
    for timeline in per_pipe_timeline.values():
        timeline.sort(key=lambda item: (item[1], seq_pos[item[0]]))

    pipe_orders = {
        pipe: list(order) for pipe, order in planned_pipe_orders.items()
    }

    memory_dependencies = []
    for (source, target), part in sorted(memory_dependency_parts.items()):
        memory_dependencies.append({
            'source': source,
            'target': target,
            'kind': '+'.join(sorted(part['kinds'])),
            'positions': sorted(part['positions']),
            'reused_bytes': part['bytes'],
            'previous_tensor_ids': sorted(
                tid for tid in part['tensor_ids'] if tid is not None),
        })

    # 多核模拟只认图依赖。把虚拟额度复用关系写成直接 op->op 边，并保留
    # metadata 便于复核；_build_graph_views 会与普通数据依赖统一解析。
    execution_graph = deepcopy(ext_graph)
    existing_edges = {
        (edge['source'], edge['target']) for edge in execution_graph['edges']
    }
    for dep in memory_dependencies:
        pair = (dep['source'], dep['target'])
        if pair in existing_edges:
            continue
        execution_graph['edges'].append({
            'source': dep['source'], 'target': dep['target'],
            'dependency': 'MEMORY_REUSE',
        })
        existing_edges.add(pair)

    # 在把结果交给多核模拟前一次性验证契约。这里验证的是生成结果本身，
    # 多核事件循环无需再重复检查内存容量或重新推导假依赖。
    scheduled_ids = {
        op_id for order in pipe_orders.values() for op_id in order
    }
    if scheduled_ids != set(op_by_id) or sum(map(len, pipe_orders.values())) != len(op_by_id):
        raise Step3SchedulingError('Step3 pipe orders do not cover every op exactly once')
    exec_in, exec_out, exec_preds, exec_succs = _build_graph_views(
        execution_graph['ops'], execution_graph['edges'])
    del exec_in, exec_out
    for target, target_preds in exec_preds.items():
        for source in target_preds:
            if op_end[source] > op_start[target]:
                raise Step3SchedulingError(
                    'Step3 dependency timing violation: {} -> {}'.format(
                        source, target))
    indegree = {op_id: len(exec_preds[op_id]) for op_id in op_by_id}
    ready = [op_id for op_id, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        source = heapq.heappop(ready)
        visited += 1
        for target in exec_succs[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if visited != len(op_by_id):
        raise Step3SchedulingError('memory reuse dependencies introduced a cycle')

    unfinished_refs = {
        tid: count for tid, count in remaining_consumers.items() if count != 0
    }
    if unfinished_refs or resident_tensors or any(memory_used.values()):
        message = (
            '[STEP3 ERROR] tensor lifetime did not close: remaining_refs={} '
            'resident_tensors={} memory_used={}'
        ).format(unfinished_refs, sorted(resident_tensors), memory_used)
        emit_error(message)
        raise Step3SchedulingError(message)

    return {
        'makespan': makespan,
        'op_start': op_start,
        'op_end': op_end,
        'per_pipe_timeline': per_pipe_timeline,
        'pipe_orders': pipe_orders,
        'execution_graph': execution_graph,
        'memory_dependencies': memory_dependencies,
        'op_status': op_status,
        'completed_count': len(op_start),
        'pred_remaining': pred_remaining,
        'in_tids': in_tids,
        'out_tids': out_tids,
        'ddr_transfer': ddr_transfer,
        'ddr_contention_log': ddr_contention_log,
        'memory_peak': memory_peak,
        'memory_used_final': memory_used,
        'memory_events': memory_events,
        'remaining_consumers': remaining_consumers,
        'resident_tensors_final': sorted(resident_tensors),
        'execution_contract_validated': True,
    }

def prepare_step3_execution(ext_graph, capacity, bandwidth):
    """运行 Step3，并返回多核评估器可直接装载的 Task 描述。

    多核阶段可以改变操作的绝对开始/结束时间，但必须保持这里给出的每条
    Pipe 顺序，并把 execution_graph 中的内存复用边当作普通完成依赖。
    输入约束及来源同step3_simulation；返回值直接引用其已验证的图和
    Pipe顺序，不做二次覆盖/局部DAG校验。跨核COPY引入的新环由评估入口检查。
    容量与带宽由调用方从 config.txt 读出后传入。
    """
    result = step3_simulation(
        ext_graph, capacity=capacity, bandwidth=bandwidth)
    graph = result['execution_graph']
    in_tids, out_tids, op_preds, op_succs = _build_graph_views(
        graph['ops'], graph['edges'])
    return {
        'graph': graph,
        'seq': graph['seq_ext'],
        'seq_pos': {op_id: index
                    for index, op_id in enumerate(graph['seq_ext'])},
        'op_by_id': {op['id']: op for op in graph['ops']},
        'tensor_by_id': {tensor['id']: tensor for tensor in graph['tensors']},
        'in_tids': in_tids,
        'out_tids': out_tids,
        'op_preds': op_preds,
        'op_succs': op_succs,
        'pipe_ops': {pipe: list(order)
                     for pipe, order in result['pipe_orders'].items()},
        'step3': result,
    }
