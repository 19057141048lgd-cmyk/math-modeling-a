"""问题 1：每个子图作为一个 Task，模拟多核、多 Pipe 执行。

阅读顺序建议：

1. ``_build_scene_a_tasks`` 按选手方案拆分原图，并补齐 Task 边界 COPY；
2. 每个 Task 依次执行 Step1、Step2、Step3，得到内存补边后的执行图；
3. ``evaluate_scene_a`` 装载 Step3 已确定的逐 Pipe 顺序；
4. 事件循环反复执行“退休、推进 Pipe、激活 Task、发射操作”；
5. 最后整理每核时间线、内存峰值和数据搬运量。

嵌套小函数只读写 ``evaluate_scene_a`` 的模拟状态，不对外提供接口。
"""

import heapq
import math
from collections import defaultdict
from evaluation_validation import validate_parameters, validate_task_order

from schedule_step1 import step1_schedule
from schedule_step2 import _build_extended_graph, step2_spill_insertion
from schedule_step3 import (
    PIPES, PIPE_SLOTS, _build_graph_views, _op_duration,
    _uses_ddr_bandwidth, op_pipe, prepare_step3_execution,
)
from stub_multicore_cut_and_schedule import derive_multicore_plan

class SceneAEvaluationError(RuntimeError):
    """场景 A 输入、Task 图或模拟状态不合法。"""

def read_scene_a_config(config_path):
    """Task 等待周期来自 config.txt 的 [multicore_scene_a]，无代码默认值。"""
    from evaluation_validation import read_required_settings
    return read_required_settings(
        config_path, 'multicore_scene_a',
        ('task_cross_core_wait_cycles', 'task_same_core_wait_cycles'))

def _original_tensor_views(graph_json):
    op_ids = {op['id'] for op in graph_json['ops']}
    producers, consumers = defaultdict(set), defaultdict(set)
    direct_edges = []
    for edge in graph_json['edges']:
        src, dst = edge['source'], edge['target']
        if src in op_ids and dst not in op_ids:
            producers[dst].add(src)
        elif src not in op_ids and dst in op_ids:
            consumers[src].add(dst)
        elif src in op_ids and dst in op_ids and src != dst:
            direct_edges.append(edge)
    return producers, consumers, direct_edges

def _copy_traffic_bytes(graph_json):
    """按 COPY_IN 输出 / COPY_OUT 输入 tensor 大小统计总搬运量。"""
    ops = graph_json.get('ops', [])
    tensors = graph_json.get('tensors', [])
    in_tids, out_tids, _, _ = _build_graph_views(
        ops, graph_json.get('edges', []))
    tensor_by_id = {tensor['id']: tensor for tensor in tensors}
    total = 0
    for op in ops:
        if op.get('op') == 'COPY_IN':
            tids = out_tids[op['id']]
        elif op.get('op') == 'COPY_OUT':
            tids = in_tids[op['id']]
        else:
            continue
        total += sum(tensor_by_id[tid]['size'] for tid in tids
                     if tid in tensor_by_id)
    return total

def _build_scene_a_tasks(graph_json, plan, bandwidth, capacity):
    """将每个子图封装为独立 Task，并为所有 Task 边界插入 DDR COPY。

    输入：原图/方案来自选手，故derive_multicore_plan负责入口校验；参数
    已由evaluate_scene_a校验。串行Task顺序是新增约束，需合并检查环。
    输出给Step1的局部图只保留原依赖、插入源/汇COPY，仍为DAG。
    """
    plan_view = derive_multicore_plan(graph_json, plan)
    validate_task_order(plan_view)
    op_by_id = {op['id']: op for op in graph_json['ops']}
    tensor_by_id = {tensor['id']: tensor for tensor in graph_json['tensors']}
    mapping = plan_view['mapping']
    producers, consumers, direct_edges = _original_tensor_views(graph_json)
    core_by_task = plan_view['core_by_subgraph']
    pred_tasks = plan_view['subgraph_preds']

    next_op_id = max([op['id'] for op in graph_json['ops']] + [0]) + 1
    next_tensor_id = max(
        [tensor['id'] for tensor in graph_json['tensors']] + [10000]) + 1
    used_ids = set(op_by_id) | set(tensor_by_id)

    def new_boundary_ids():
        # 原图两类ID已唯一；新ID同时避开两类已占用空间，不能只取各自max。
        nonlocal next_op_id, next_tensor_id
        while next_tensor_id in used_ids:
            next_tensor_id += 1
        ddr_id = next_tensor_id
        used_ids.add(ddr_id)
        next_tensor_id += 1
        while next_op_id in used_ids:
            next_op_id += 1
        copy_id = next_op_id
        used_ids.add(copy_id)
        next_op_id += 1
        return ddr_id, copy_id
    tasks = {}
    cross_task_traffic = 0
    task_graph_copy_traffic = 0
    spill_copy_traffic = 0

    for task_id in plan_view['subgraph_ids']:
        task_op_ids = set(plan_view['nodes_by_subgraph'][task_id])
        ops = [dict(op_by_id[op_id]) for op_id in sorted(task_op_ids)]
        tensors, edges = [], []
        local_tensor_ids = set()

        touched_tensors = sorted(
            tensor_id for tensor_id in tensor_by_id
            if producers.get(tensor_id, set()) & task_op_ids
            or consumers.get(tensor_id, set()) & task_op_ids)
        for tensor_id in touched_tensors:
            tensor = dict(tensor_by_id[tensor_id])
            local_producers = producers.get(tensor_id, set()) & task_op_ids
            local_consumers = consumers.get(tensor_id, set()) & task_op_ids
            eligible_producers = {op for op in producers.get(tensor_id, ()) if op in mapping}
            eligible_consumers = {op for op in consumers.get(tensor_id, ()) if op in mapping}
            has_original_copy_out = any(
                op_by_id[op_id].get('op') == 'COPY_OUT'
                for op_id in consumers.get(tensor_id, ())
                if op_id in op_by_id)
            input_boundary = bool(local_consumers) and not bool(local_producers)
            output_boundary = bool(local_producers) and (
                has_original_copy_out or not eligible_consumers
                or bool(eligible_consumers - task_op_ids))

            # Task 内 Tensor 必须在私有缓存；原 DDR Tensor 仅作为边界副本。
            if tensor.get('pos') == 'DDR':
                tensor['pos'] = 'UB'
            tensors.append(tensor)
            local_tensor_ids.add(tensor_id)
            for producer_id in sorted(local_producers):
                edges.append({'source': producer_id, 'target': tensor_id})
            for consumer_id in sorted(local_consumers):
                edges.append({'source': tensor_id, 'target': consumer_id})

            if input_boundary:
                ddr_id, copy_id = new_boundary_ids()
                tensors.append({'id': ddr_id, 'pos': 'DDR', 'size': tensor['size']})
                ops.append({'id': copy_id, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2',
                            'cycles': max(1, math.ceil(tensor['size'] / bandwidth))})
                edges.extend([
                    {'source': ddr_id, 'target': copy_id},
                    {'source': copy_id, 'target': tensor_id},
                ])
            if output_boundary:
                ddr_id, copy_id = new_boundary_ids()
                tensors.append({'id': ddr_id, 'pos': 'DDR', 'size': tensor['size']})
                ops.append({'id': copy_id, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3',
                            'cycles': max(1, math.ceil(tensor['size'] / bandwidth))})
                edges.extend([
                    {'source': tensor_id, 'target': copy_id},
                    {'source': copy_id, 'target': ddr_id},
                ])
                remote_consumer_tasks = {
                    mapping[op_id] for op_id in eligible_consumers
                    if mapping[op_id] != task_id
                }
                cross_task_traffic += tensor['size'] * len(remote_consumer_tasks)

        for edge in direct_edges:
            if edge['source'] in task_op_ids and edge['target'] in task_op_ids:
                edges.append(dict(edge))

        graph = {'ops': ops, 'tensors': tensors, 'edges': edges}
        task_graph_copy_traffic += _copy_traffic_bytes(graph)
        seq = step1_schedule(graph)
        result2 = step2_spill_insertion(graph, seq, capacity=capacity)
        spill_copy_traffic += sum(
            spill['size'] * (1 + int(spill['spill_out_copies_data']))
            for spill in result2['spill_records'])
        ext_graph = _build_extended_graph(graph, result2)
        prepared = prepare_step3_execution(
            ext_graph, capacity=capacity, bandwidth=bandwidth)
        prepared.update({
            'task_id': task_id,
            'core_id': core_by_task[task_id],
            'pred_tasks': pred_tasks[task_id],
        })
        tasks[task_id] = prepared
    original_copy_traffic = _copy_traffic_bytes(graph_json)
    partition_added_traffic = task_graph_copy_traffic - original_copy_traffic
    traffic = {
        'original_graph_copy_bytes': original_copy_traffic,
        'scheduled_copy_bytes': task_graph_copy_traffic + spill_copy_traffic,
        'added_copy_bytes': partition_added_traffic + spill_copy_traffic,
        'partition_added_copy_bytes': partition_added_traffic,
        'spill_added_copy_bytes': spill_copy_traffic,
    }
    return tasks, cross_task_traffic, traffic, plan_view

def evaluate_scene_a(graph_json, plan, bandwidth, capacity,
                     cross_core_wait, same_core_wait, max_iter=1000000):
    """执行场景 A 多核事件模拟并返回 makespan 与完整时间线。

    带宽、容量与两类等待周期都由 CLI 从 config.txt 读出后传入，本函数不再
    提供配置默认值；每条 Pipe 同一时刻只有 PIPE_SLOTS 个在飞指令，不对外开放。
    """
    capacity = dict(capacity)
    validate_parameters(bandwidth, capacity, max_iter,
                        cross_core_wait=cross_core_wait, same_core_wait=same_core_wait)
    tasks, cross_task_traffic, data_movement, plan_view = _build_scene_a_tasks(
        graph_json, plan, bandwidth, capacity)
    num_cores = plan_view['num_cores']
    core_orders = plan_view['core_orders']
    task_status = {task_id: 'waiting' for task_id in tasks}
    task_start, task_end = {}, {}
    core_index = {core_id: 0 for core_id in range(num_cores)}
    core_active_task = {core_id: None for core_id in range(num_cores)}
    core_previous_end = {core_id: None for core_id in range(num_cores)}
    executors = {
        (core_id, pipe): []
        for core_id in range(num_cores) for pipe in PIPES
    }
    issue_queues = {
        (core_id, pipe): []
        for core_id in range(num_cores) for pipe in PIPES
    }
    op_status, pred_remaining, op_start, op_end = {}, {}, {}, {}
    ddr_remaining_work, ddr_contention_log = {}, []
    ddr_last_update = 0

    def key(task_id, op_id):
        return (task_id, op_id)

    def init_pipe_queues(task):
        """装载 Step3 已确定的逐 Pipe 顺序。"""
        task['pipe_cursor'] = {pipe: 0 for pipe in task['pipe_ops']}

    def advance_ddr_work(now):
        nonlocal ddr_last_update
        elapsed = now - ddr_last_update
        while elapsed > 1e-9:
            active = [item for item, work in ddr_remaining_work.items() if work > 1e-9]
            if not active:
                break
            min_work = min(ddr_remaining_work[item] for item in active)
            finish_delta = min_work * len(active)
            if finish_delta >= elapsed - 1e-9:
                share = elapsed / len(active)
                for item in active:
                    ddr_remaining_work[item] = max(0.0, ddr_remaining_work[item] - share)
                break
            for item in active:
                ddr_remaining_work[item] = max(0.0, ddr_remaining_work[item] - min_work)
            elapsed -= finish_delta
        ddr_last_update = now

    def reschedule_ddr(now):
        if not ddr_remaining_work:
            return {}
        ordered = sorted((max(0.0, work), item)
                         for item, work in ddr_remaining_work.items())
        projected, cursor, previous, active_count = {}, float(now), 0.0, len(ordered)
        i = 0
        while i < len(ordered):
            work = ordered[i][0]
            cursor += (work - previous) * active_count
            j = i
            while j < len(ordered) and abs(ordered[j][0] - work) <= 1e-9:
                projected[ordered[j][1]] = int(math.ceil(cursor - 1e-9))
                j += 1
            active_count -= j - i
            previous = work
            i = j
        for item, end in projected.items():
            op_end[item] = end
        for executor_key, running in executors.items():
            executors[executor_key] = [
                (item, projected.get(item, end)) for item, end in running]
        return projected

    def queue_if_ready(task_id, op_id):
        item = key(task_id, op_id)
        if op_status[item] != 'pending' or pred_remaining[item] != 0:
            return
        task = tasks[task_id]
        pipe = op_pipe(task['op_by_id'][op_id])
        order = task['pipe_ops'][pipe]
        cursor = task['pipe_cursor'][pipe]
        if cursor >= len(order) or order[cursor] != op_id:
            return
        op_status[item] = 'ready'
        heapq.heappush(issue_queues[(task['core_id'], pipe)],
                       (task['seq_pos'][op_id], item))

    def activate_task(task_id, now):
        task = tasks[task_id]
        init_pipe_queues(task)
        task_status[task_id] = 'active'
        task_start[task_id] = now
        core_active_task[task['core_id']] = task_id
        for op_id in task['seq']:
            item = key(task_id, op_id)
            op_status[item] = 'pending'
            pred_remaining[item] = len(task['op_preds'][op_id])
        for op_id in task['seq']:
            queue_if_ready(task_id, op_id)

    def advance_pipe(task_id, op_id):
        """操作完成后推进所在 Pipe，并尝试唤醒下一操作。"""
        task = tasks[task_id]
        pipe = op_pipe(task['op_by_id'][op_id])
        cursor = task['pipe_cursor'][pipe]
        order = task['pipe_ops'][pipe]
        task['pipe_cursor'][pipe] += 1
        if cursor + 1 < len(order):
            queue_if_ready(task_id, order[cursor + 1])

    def task_release_time(task_id):
        task = tasks[task_id]
        core_id = task['core_id']
        if any(task_status[pred] != 'done' for pred in task['pred_tasks']):
            return None
        release = 0
        if core_previous_end[core_id] is not None:
            release = core_previous_end[core_id] + same_core_wait
        for pred in task['pred_tasks']:
            if tasks[pred]['core_id'] != core_id:
                release = max(release, task_end[pred] + cross_core_wait)
        return release

    def activate_ready_tasks(now):
        changed = False
        for core_id in range(num_cores):
            if core_active_task[core_id] is not None:
                continue
            order = core_orders.get(core_id, [])
            if core_index[core_id] >= len(order):
                continue
            task_id = order[core_index[core_id]]
            release = task_release_time(task_id)
            if release is not None and release <= now:
                activate_task(task_id, now)
                changed = True
        return changed

    def retire(now):
        advance_ddr_work(now)
        retired_ddr = False
        for executor_key, running in list(executors.items()):
            keep = []
            for item, end in running:
                if end > now:
                    keep.append((item, end))
                    continue
                task_id, op_id = item
                op_status[item] = 'done'
                advance_pipe(task_id, op_id)
                if item in ddr_remaining_work:
                    ddr_remaining_work.pop(item)
                    retired_ddr = True
                for succ_id in tasks[task_id]['op_succs'][op_id]:
                    succ_item = key(task_id, succ_id)
                    pred_remaining[succ_item] -= 1
                    queue_if_ready(task_id, succ_id)
            executors[executor_key] = keep
        if retired_ddr:
            reschedule_ddr(now)

        for task_id, status in list(task_status.items()):
            if status != 'active':
                continue
            task_items = [key(task_id, op_id) for op_id in tasks[task_id]['seq']]
            if all(op_status[item] == 'done' for item in task_items):
                core_id = tasks[task_id]['core_id']
                task_status[task_id] = 'done'
                task_end[task_id] = now
                core_active_task[core_id] = None
                core_previous_end[core_id] = now
                core_index[core_id] += 1

    def issue(now):
        issued_in_pass = True
        while issued_in_pass:
            issued_in_pass = False
            for core_id in range(num_cores):
                for pipe in PIPES:
                    executor_key = (core_id, pipe)
                    queue = issue_queues[executor_key]
                    while len(executors[executor_key]) < PIPE_SLOTS:
                        if not queue:
                            break
                        _, item = heapq.heappop(queue)
                        issued_in_pass = True
                        task_id, op_id = item
                        task = tasks[task_id]
                        op = task['op_by_id'][op_id]
                        duration = _op_duration(
                            op, task['in_tids'], task['out_tids'],
                            task['tensor_by_id'], bandwidth)
                        op_status[item] = 'running'
                        op_start[item] = now
                        op_end[item] = now + duration
                        executors[executor_key].append((item, op_end[item]))
                        if _uses_ddr_bandwidth(
                                op, task['in_tids'], task['out_tids'],
                                task['tensor_by_id']):
                            advance_ddr_work(now)
                            ddr_remaining_work[item] = float(duration)
                            projected = reschedule_ddr(now)
                            ddr_contention_log.append({
                                'time': now,
                                'issued': {'task_id': task_id, 'op_id': op_id,
                                           'core_id': core_id, 'pipe': pipe},
                                'active_count': len(ddr_remaining_work),
                                'projected_ends': [
                                    {'task_id': active[0], 'op_id': active[1], 'end': end}
                                    for active, end in sorted(projected.items())
                                ],
                            })

    now = 0
    for iteration in range(max_iter):
        retire(now)
        activate_ready_tasks(now)
        issue(now)
        if all(status == 'done' for status in task_status.values()):
            break
        next_times = [end for running in executors.values() for _, end in running]
        for core_id in range(num_cores):
            if core_active_task[core_id] is not None:
                continue
            order = core_orders.get(core_id, [])
            if core_index[core_id] < len(order):
                release = task_release_time(order[core_index[core_id]])
                if release is not None and release > now:
                    next_times.append(release)
        if not next_times:
            waiting = sorted(task_id for task_id, status in task_status.items()
                             if status != 'done')
            raise SceneAEvaluationError(
                'multicore scheduler deadlock at t={}; waiting_tasks={}'.format(
                    now, waiting))
        next_now = min(next_times)
        if next_now <= now:
            raise SceneAEvaluationError(
                'multicore scheduler made no progress at t={}'.format(now))
        now = next_now
    else:
        raise SceneAEvaluationError('max_iter exceeded')

    per_core_timeline = []
    for core_id in range(num_cores):
        task_entries = []
        for task_id in core_orders.get(core_id, []):
            task_entries.append({
                'task_id': task_id, 'subgraph_id': task_id,
                'start': task_start[task_id], 'end': task_end[task_id],
                'duration': task_end[task_id] - task_start[task_id],
            })
        op_entries = []
        for item, start in op_start.items():
            task_id, op_id = item
            if tasks[task_id]['core_id'] != core_id:
                continue
            op = tasks[task_id]['op_by_id'][op_id]
            op_entries.append({
                'task_id': task_id, 'op_id': op_id, 'op': op['op'],
                'pipe': op_pipe(op),
                'start': start, 'end': op_end[item],
                'duration': op_end[item] - start,
            })
        op_entries.sort(key=lambda entry: (entry['start'], entry['task_id'], entry['op_id']))
        subgraph_entries = []
        for task_id in core_orders.get(core_id, []):
            subgraph_ops = [entry for entry in op_entries
                            if entry['task_id'] == task_id]
            if not subgraph_ops:
                raise SceneAEvaluationError(
                    'subgraph {} has no scheduled op'.format(task_id))
            start = min(entry['start'] for entry in subgraph_ops)
            end = max(entry['end'] for entry in subgraph_ops)
            subgraph_entries.append({
                'subgraph_id': task_id,
                'start': start, 'end': end, 'duration': end - start,
            })
        per_core_timeline.append({
            'core_id': core_id, 'tasks': task_entries,
            'subgraphs': subgraph_entries, 'ops': op_entries})

    makespan = max(task_end.values(), default=0)
    memory_peak_by_core = {
        core_id: {
            pos: max((tasks[task_id]['step3']['memory_peak'][pos]
                      for task_id in core_orders.get(core_id, [])), default=0)
            for pos in capacity
        }
        for core_id in range(num_cores)
    }
    return {
        'scene': 'A',
        'makespan': makespan,
        'num_cores': num_cores,
        'bandwidth_bytes_per_cycle': bandwidth,
        'capacity_bytes': dict(capacity),
        'memory_peak_by_core': memory_peak_by_core,
        'step3_by_task': {
            task_id: {
                'local_makespan': task['step3']['makespan'],
                'memory_dependency_count': len(
                    task['step3']['memory_dependencies']),
                'pipe_op_counts': {
                    pipe: len(order) for pipe, order in task['pipe_ops'].items()
                },
            }
            for task_id, task in tasks.items()
        },
        'task_cross_core_wait_cycles': cross_core_wait,
        'task_same_core_wait_cycles': same_core_wait,
        'cross_task_traffic': cross_task_traffic,
        'data_movement_bytes': data_movement,
        'task_dependencies': [
            {'source': source, 'target': target}
            for source, target in plan_view['dependency_pairs']
        ],
        'per_core_timeline': per_core_timeline,
        'ddr_contention_log': ddr_contention_log,
    }

if __name__ == '__main__':
    from contest_io import run_problem_cli
    raise SystemExit(run_problem_cli(1))
