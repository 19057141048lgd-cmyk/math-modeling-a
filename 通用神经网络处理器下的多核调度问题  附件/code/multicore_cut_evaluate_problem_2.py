"""问题 2：每个核心合并成一个 Task，模拟多核、多 Pipe 执行。

同核子图通过片上 tensor 直接通信；跨核边插入 COPY_OUT/COPY_IN，
目标 COPY_IN 最早可在源 COPY_OUT 结束后 delay cycles 发射。

主流程是：构造每核 Task → 插入跨核 COPY → Step1/2/3 核内调度 →
装载 Step3 的内存补边图与 Pipe 顺序 → 事件模拟 → 汇总时间线。事件循环只做
三件事：完成到期操作、释放到期的跨核 COPY_IN、发射当前可执行操作。
"""

import heapq
import math
from collections import defaultdict

from multicore_cut_evaluate_problem_1 import (
    _copy_traffic_bytes, _original_tensor_views,
)
from schedule_step1 import _check_topo, step1_schedule
from schedule_step2 import _build_extended_graph, step2_spill_insertion
from schedule_step3 import (
    PIPES, PIPE_SLOTS, _op_duration, _uses_ddr_bandwidth, op_pipe,
    prepare_step3_execution,
)
from stub_multicore_cut_and_schedule import derive_multicore_plan
from evaluation_validation import validate_execution, validate_parameters

class SceneBEvaluationError(RuntimeError):
    """场景 B 输入、跨核通信图或模拟状态不合法。"""

def read_scene_b_config(config_path):
    """跨核 COPY 等待来自 config.txt 的 [multicore_scene_b]，无代码默认值。"""
    from evaluation_validation import read_required_settings
    return read_required_settings(
        config_path, 'multicore_scene_b', ('cross_core_copy_delay_cycles',))

def _append_tensor(task_data, tensor):
    task_data['tensors'].setdefault(tensor['id'], dict(tensor))

def _append_op(task_data, op, subgraph_id):
    task_data['ops'].append(dict(op))
    task_data['op_subgraph'][op['id']] = subgraph_id

def _prioritize_task_seq(graph, raw_seq, op_subgraph, subgraph_order):
    """依照核内子图调度序分桶，桶内保留 Step1 顺序。

    输入raw_seq的完整性/拓扑性由Step1保证，稳定排序保持覆盖性。
    子图优先级是选手提供的新顺序，可能破坏拓扑，故只校验这一新约束。
    """
    rank = {subgraph_id: index
            for index, subgraph_id in enumerate(subgraph_order)}
    fallback = len(rank)
    seq = sorted(
        raw_seq,
        key=lambda op_id: rank.get(op_subgraph.get(op_id), fallback),
    )
    if not _check_topo(graph, seq):
        raise SceneBEvaluationError(
            'subgraph priority order violates an intra-core dependency')
    return seq

def _build_scene_b_tasks(graph_json, plan, bandwidth, capacity):
    """每核合并为一个 Task，仅对跨核边插入 DDR COPY 对。

    原图/方案由本函数开头的derive_multicore_plan进行入口校验，参数由
    evaluate_scene_b校验。局部图保留本核依赖并插入源/汇COPY，仍为DAG；
    子图重排的拓扑性由_prioritize_task_seq保证，Step2/3无需重复检查。
    """
    plan_view = derive_multicore_plan(graph_json, plan)
    num_cores = plan_view['num_cores']
    op_by_id = {op['id']: op for op in graph_json['ops']}
    tensor_by_id = {tensor['id']: tensor for tensor in graph_json['tensors']}
    mapping = plan_view['mapping']
    core_by_subgraph = plan_view['core_by_subgraph']
    core_by_op = {op_id: core_by_subgraph[subgraph_id]
                  for op_id, subgraph_id in mapping.items()}
    core_orders = plan_view['core_orders']
    producers, consumers, direct_edges = _original_tensor_views(graph_json)
    tasks_data = {
        core_id: {
            'ops': [], 'tensors': {}, 'edges': [], 'op_subgraph': {},
            'subgraph_order': core_orders.get(core_id, []),
        }
        for core_id in range(num_cores)
    }
    for op_id, subgraph_id in mapping.items():
        _append_op(tasks_data[core_by_op[op_id]], op_by_id[op_id], subgraph_id)

    max_node_id = max(
        [op['id'] for op in graph_json['ops']]
        + [tensor['id'] for tensor in graph_json['tensors']] + [0])
    next_id = max_node_id + 1
    cross_links = []
    cross_task_traffic = 0
    task_graph_copy_traffic = 0
    spill_copy_traffic = 0

    def new_id():
        nonlocal next_id
        value = next_id
        next_id += 1
        return value

    def add_copy_in(core_id, local_tid, size, subgraph_id, ddr_tid=None):
        ddr_tid = new_id() if ddr_tid is None else ddr_tid
        copy_id = new_id()
        _append_tensor(tasks_data[core_id], {
            'id': ddr_tid, 'pos': 'DDR', 'size': size})
        _append_op(tasks_data[core_id], {
            'id': copy_id, 'op': 'COPY_IN', 'pipe': 'PIPE_MTE2',
            'cycles': max(1, math.ceil(size / bandwidth)),
        }, subgraph_id)
        tasks_data[core_id]['edges'].extend([
            {'source': ddr_tid, 'target': copy_id},
            {'source': copy_id, 'target': local_tid},
        ])
        return copy_id, ddr_tid

    def add_copy_out(core_id, local_tid, size, subgraph_id, ddr_tid=None):
        ddr_tid = new_id() if ddr_tid is None else ddr_tid
        copy_id = new_id()
        _append_tensor(tasks_data[core_id], {
            'id': ddr_tid, 'pos': 'DDR', 'size': size})
        _append_op(tasks_data[core_id], {
            'id': copy_id, 'op': 'COPY_OUT', 'pipe': 'PIPE_MTE3',
            'cycles': max(1, math.ceil(size / bandwidth)),
        }, subgraph_id)
        tasks_data[core_id]['edges'].extend([
            {'source': local_tid, 'target': copy_id},
            {'source': copy_id, 'target': ddr_tid},
        ])
        return copy_id, ddr_tid

    for tensor_id in sorted(tensor_by_id):
        tensor = tensor_by_id[tensor_id]
        eligible_producers = sorted(op for op in producers.get(tensor_id, ())
                                    if op in mapping)
        eligible_consumers = sorted(op for op in consumers.get(tensor_id, ())
                                    if op in mapping)
        producer_cores = sorted({core_by_op[op] for op in eligible_producers})
        consumer_cores = sorted({core_by_op[op] for op in eligible_consumers})
        touched_cores = sorted(set(producer_cores) | set(consumer_cores))
        if not touched_cores:
            continue
        local_tensor = dict(tensor)
        if local_tensor.get('pos') == 'DDR':
            local_tensor['pos'] = 'UB'
        for core_id in touched_cores:
            _append_tensor(tasks_data[core_id], local_tensor)
            for op_id in eligible_producers:
                if core_by_op[op_id] == core_id:
                    tasks_data[core_id]['edges'].append(
                        {'source': op_id, 'target': tensor_id})
            for op_id in eligible_consumers:
                if core_by_op[op_id] == core_id:
                    tasks_data[core_id]['edges'].append(
                        {'source': tensor_id, 'target': op_id})

        # 图输入：每个消费核各自从 DDR 读入一次。
        if eligible_consumers and not eligible_producers:
            for dst_core in consumer_cores:
                dst_ops = [op for op in eligible_consumers
                           if core_by_op[op] == dst_core]
                dst_subgraph = min(
                    (mapping[op] for op in dst_ops),
                    key=lambda sg: core_orders[dst_core].index(sg))
                add_copy_in(dst_core, tensor_id, tensor['size'], dst_subgraph)

        # 图输出：保留对 DDR 的最终写回。
        has_original_copy_out = any(
            op_by_id[op_id].get('op') == 'COPY_OUT'
            for op_id in consumers.get(tensor_id, ()) if op_id in op_by_id)
        if eligible_producers and (has_original_copy_out or not eligible_consumers):
            for src_core in producer_cores:
                src_ops = [op for op in eligible_producers
                           if core_by_op[op] == src_core]
                src_subgraph = max(
                    (mapping[op] for op in src_ops),
                    key=lambda sg: core_orders[src_core].index(sg))
                add_copy_out(src_core, tensor_id, tensor['size'], src_subgraph)

        # 跨核 tensor：每个实际 source-core -> target-core 连接一对 COPY。
        for src_core in producer_cores:
            src_ops = [op for op in eligible_producers
                       if core_by_op[op] == src_core]
            src_subgraph = max(
                (mapping[op] for op in src_ops),
                key=lambda sg: core_orders[src_core].index(sg))
            for dst_core in consumer_cores:
                if src_core == dst_core:
                    continue
                dst_ops = [op for op in eligible_consumers
                           if core_by_op[op] == dst_core]
                dst_subgraph = min(
                    (mapping[op] for op in dst_ops),
                    key=lambda sg: core_orders[dst_core].index(sg))
                ddr_tid = new_id()
                out_id, _ = add_copy_out(
                    src_core, tensor_id, tensor['size'], src_subgraph, ddr_tid)
                in_id, _ = add_copy_in(
                    dst_core, tensor_id, tensor['size'], dst_subgraph, ddr_tid)
                cross_links.append({
                    'tensor_id': tensor_id, 'size': tensor['size'],
                    'source_core': src_core, 'target_core': dst_core,
                    'source_copy_out_id': out_id,
                    'target_copy_in_id': in_id,
                })
                cross_task_traffic += tensor['size']

    # 直接 op-op 边：同核保留，跨核转成显式 COPY 对。
    for edge in direct_edges:
        src, dst = edge['source'], edge['target']
        if src not in mapping or dst not in mapping:
            continue
        src_core, dst_core = core_by_op[src], core_by_op[dst]
        if src_core == dst_core:
            tasks_data[src_core]['edges'].append(dict(edge))
            continue
        size = max(0, int(edge.get('data_size', 0)))
        local_tid, ddr_tid = new_id(), new_id()
        local_tensor = {'id': local_tid, 'pos': 'UB', 'size': size}
        _append_tensor(tasks_data[src_core], local_tensor)
        _append_tensor(tasks_data[dst_core], local_tensor)
        tasks_data[src_core]['edges'].append({'source': src, 'target': local_tid})
        tasks_data[dst_core]['edges'].append({'source': local_tid, 'target': dst})
        out_id, _ = add_copy_out(
            src_core, local_tid, size, mapping[src], ddr_tid)
        in_id, _ = add_copy_in(
            dst_core, local_tid, size, mapping[dst], ddr_tid)
        cross_links.append({
            'tensor_id': local_tid, 'size': size,
            'source_core': src_core, 'target_core': dst_core,
            'source_copy_out_id': out_id, 'target_copy_in_id': in_id,
        })
        cross_task_traffic += size

    tasks = {}
    for core_id in range(num_cores):
        data = tasks_data[core_id]
        graph = {
            'ops': data['ops'],
            'tensors': list(data['tensors'].values()),
            'edges': data['edges'],
        }
        task_graph_copy_traffic += _copy_traffic_bytes(graph)
        raw_seq = step1_schedule(graph) if graph['ops'] else []
        seq = _prioritize_task_seq(
            graph, raw_seq, data['op_subgraph'], data['subgraph_order'])
        result2 = step2_spill_insertion(
            graph, seq, capacity=capacity) if seq else {
                'new_ops': [], 'new_tensors': [], 'new_edges': [],
                'spill_records': [], 'seq_ext': []}
        spill_copy_traffic += sum(
            spill['size'] * (1 + int(spill['spill_out_copies_data']))
            for spill in result2.get('spill_records', []))
        ext_op_subgraph = dict(data['op_subgraph'])
        for spill in result2.get('spill_records', []):
            next_subgraph = ext_op_subgraph.get(spill.get('next_use_op'))
            previous_subgraph = ext_op_subgraph.get(spill.get('prev_use_op'))
            if spill['spill_out_id'] is not None:
                ext_op_subgraph[spill['spill_out_id']] = (
                    previous_subgraph if previous_subgraph is not None
                    else next_subgraph)
            ext_op_subgraph[spill['spill_in_id']] = next_subgraph
        ext_graph = _build_extended_graph(graph, result2)
        prepared = prepare_step3_execution(
            ext_graph, capacity=capacity, bandwidth=bandwidth)
        prepared.update({
            'task_id': core_id, 'core_id': core_id,
            'subgraph_ids': data['subgraph_order'],
            'op_subgraph': ext_op_subgraph,
        })
        tasks[core_id] = prepared
    original_copy_traffic = _copy_traffic_bytes(graph_json)
    partition_added_traffic = task_graph_copy_traffic - original_copy_traffic
    traffic = {
        'original_graph_copy_bytes': original_copy_traffic,
        'scheduled_copy_bytes': task_graph_copy_traffic + spill_copy_traffic,
        'added_copy_bytes': partition_added_traffic + spill_copy_traffic,
        'partition_added_copy_bytes': partition_added_traffic,
        'spill_added_copy_bytes': spill_copy_traffic,
    }
    return tasks, cross_links, cross_task_traffic, traffic, plan_view

def evaluate_scene_b(graph_json, plan, bandwidth, capacity,
                     cross_core_copy_delay, max_iter=1000000):
    """执行场景 B 多核事件模拟，返回 makespan 和时间线。

    带宽、容量与跨核等待都由 CLI 从 config.txt 读出后传入，本函数不再提供
    配置默认值；每条 Pipe 同一时刻只有 PIPE_SLOTS 个在飞指令，不对外开放。
    """
    capacity = dict(capacity)
    validate_parameters(bandwidth, capacity, max_iter,
                        cross_core_copy_delay=cross_core_copy_delay)
    (tasks, cross_links, cross_task_traffic,
     data_movement, plan_view) = _build_scene_b_tasks(
         graph_json, plan, bandwidth, capacity)
    validate_execution(tasks, cross_links)
    num_cores = plan_view['num_cores']
    executors = {
        (core_id, pipe): []
        for core_id in range(num_cores) for pipe in PIPES
    }
    issue_queues = {
        (core_id, pipe): []
        for core_id in range(num_cores) for pipe in PIPES
    }
    op_status, pred_remaining, op_start, op_end = {}, {}, {}, {}
    external_preds, external_succs = defaultdict(list), defaultdict(list)
    external_release_heap, release_scheduled = [], set()
    ddr_remaining_work = {}
    ddr_last_update = 0

    def key(core_id, op_id):
        return (core_id, op_id)

    def init_pipe_queues(task):
        """装载 Step3 已确定的逐 Pipe 顺序。"""
        task['pipe_cursor'] = {pipe: 0 for pipe in task['pipe_ops']}

    for task in tasks.values():
        init_pipe_queues(task)

    for link in cross_links:
        source = key(link['source_core'], link['source_copy_out_id'])
        target = key(link['target_core'], link['target_copy_in_id'])
        external_preds[target].append(source)
        external_succs[source].append(target)

    def advance_ddr_work(now):
        nonlocal ddr_last_update
        elapsed = now - ddr_last_update
        while elapsed > 1e-9:
            active = [item for item, work in ddr_remaining_work.items()
                      if work > 1e-9]
            if not active:
                break
            min_work = min(ddr_remaining_work[item] for item in active)
            finish_delta = min_work * len(active)
            if finish_delta >= elapsed - 1e-9:
                share = elapsed / len(active)
                for item in active:
                    ddr_remaining_work[item] = max(
                        0.0, ddr_remaining_work[item] - share)
                break
            for item in active:
                ddr_remaining_work[item] = max(
                    0.0, ddr_remaining_work[item] - min_work)
            elapsed -= finish_delta
        ddr_last_update = now

    def reschedule_ddr(now):
        if not ddr_remaining_work:
            return {}
        ordered = sorted((max(0.0, work), item)
                         for item, work in ddr_remaining_work.items())
        projected, cursor, previous = {}, float(now), 0.0
        active_count, i = len(ordered), 0
        while i < len(ordered):
            work = ordered[i][0]
            cursor += (work - previous) * active_count
            j = i
            while j < len(ordered) and abs(ordered[j][0] - work) <= 1e-9:
                projected[ordered[j][1]] = int(math.ceil(cursor - 1e-9))
                j += 1
            active_count -= j - i
            previous, i = work, j
        for item, end in projected.items():
            op_end[item] = end
        for executor_key, running in executors.items():
            executors[executor_key] = [
                (item, projected.get(item, end)) for item, end in running]
        return projected

    def external_release(item):
        preds = external_preds.get(item, ())
        if any(op_status.get(pred) != 'done' for pred in preds):
            return None
        return max((op_end[pred] + cross_core_copy_delay for pred in preds),
                   default=0)

    def queue_if_ready(item, now):
        if op_status[item] != 'pending' or pred_remaining[item] != 0:
            return
        core_id, op_id = item
        task = tasks[core_id]
        pipe = op_pipe(task['op_by_id'][op_id])
        order = task['pipe_ops'][pipe]
        cursor = task['pipe_cursor'][pipe]
        if cursor >= len(order) or order[cursor] != op_id:
            return
        release = external_release(item)
        if release is None:
            return
        if release > now:
            if item not in release_scheduled:
                heapq.heappush(external_release_heap, (release, item))
                release_scheduled.add(item)
            return
        release_scheduled.discard(item)
        op_status[item] = 'ready'
        heapq.heappush(issue_queues[(core_id, pipe)],
                       (task['seq_pos'][op_id], item))

    def advance_pipe(core_id, op_id, now):
        """完成 Pipe 队首操作，然后尝试唤醒下一操作。"""
        task = tasks[core_id]
        pipe = op_pipe(task['op_by_id'][op_id])
        cursor = task['pipe_cursor'][pipe]
        order = task['pipe_ops'][pipe]
        task['pipe_cursor'][pipe] += 1
        if cursor + 1 < len(order):
            queue_if_ready(key(core_id, order[cursor + 1]), now)

    for core_id, task in tasks.items():
        for op_id in task['seq']:
            item = key(core_id, op_id)
            op_status[item] = 'pending'
            pred_remaining[item] = len(task['op_preds'][op_id])
    for core_id, task in tasks.items():
        for op_id in task['seq']:
            queue_if_ready(key(core_id, op_id), 0)

    remaining_ops = len(op_status)

    def retire(now):
        nonlocal remaining_ops
        advance_ddr_work(now)
        retired_ddr = False
        retired_items = []
        for executor_key, running in list(executors.items()):
            keep = []
            for item, end in running:
                if end > now:
                    keep.append((item, end))
                    continue
                op_status[item] = 'done'
                remaining_ops -= 1
                retired_items.append(item)
                if item in ddr_remaining_work:
                    ddr_remaining_work.pop(item)
                    retired_ddr = True
                core_id, op_id = item
                advance_pipe(core_id, op_id, now)
                for succ_id in tasks[core_id]['op_succs'][op_id]:
                    succ = key(core_id, succ_id)
                    pred_remaining[succ] -= 1
                    queue_if_ready(succ, now)
            executors[executor_key] = keep
        if retired_ddr:
            reschedule_ddr(now)
        for item in retired_items:
            for target in external_succs.get(item, ()):
                queue_if_ready(target, now)

    def issue(now):
        issued_in_pass = True
        while issued_in_pass:
            issued_in_pass = False
            for core_id in range(num_cores):
                task = tasks[core_id]
                for pipe in PIPES:
                    executor_key = (core_id, pipe)
                    queue = issue_queues[executor_key]
                    while len(executors[executor_key]) < PIPE_SLOTS:
                        if not queue:
                            break
                        _, item = heapq.heappop(queue)
                        issued_in_pass = True
                        _, op_id = item
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
                            reschedule_ddr(now)

    now = 0
    for _ in range(max_iter):
        retire(now)
        while external_release_heap and external_release_heap[0][0] <= now:
            _, item = heapq.heappop(external_release_heap)
            release_scheduled.discard(item)
            queue_if_ready(item, now)
        issue(now)
        if remaining_ops == 0:
            break
        next_times = [end for running in executors.values()
                      for _, end in running]
        if external_release_heap:
            next_times.append(external_release_heap[0][0])
        if not next_times:
            waiting = sorted(item for item, status in op_status.items()
                             if status != 'done')
            raise SceneBEvaluationError(
                'multicore scheduler deadlock at t={}; waiting_ops={}'.format(
                    now, waiting[:50]))
        next_now = min(next_times)
        if next_now <= now:
            raise SceneBEvaluationError(
                'multicore scheduler made no progress at t={}'.format(now))
        now = next_now
    else:
        raise SceneBEvaluationError('max_iter exceeded')

    per_core_timeline = []
    for core_id in range(num_cores):
        task = tasks[core_id]
        starts = [op_start[key(core_id, op_id)] for op_id in task['seq']]
        ends = [op_end[key(core_id, op_id)] for op_id in task['seq']]
        task_start = min(starts, default=0)
        task_end = max(ends, default=0)
        op_entries = []
        for op_id in task['seq']:
            item = key(core_id, op_id)
            op = task['op_by_id'][op_id]
            op_entries.append({
                'task_id': core_id, 'op_id': op_id, 'op': op['op'],
                'subgraph_id': task['op_subgraph'].get(op_id),
                'pipe': op_pipe(op),
                'start': op_start[item], 'end': op_end[item],
                'duration': op_end[item] - op_start[item],
            })
        op_entries.sort(key=lambda entry: (entry['start'], entry['op_id']))
        ops_by_subgraph = defaultdict(list)
        for entry in op_entries:
            ops_by_subgraph[entry['subgraph_id']].append(entry)
        subgraph_entries = []
        for subgraph_id in task['subgraph_ids']:
            subgraph_ops = ops_by_subgraph.get(subgraph_id, ())
            if not subgraph_ops:
                raise SceneBEvaluationError(
                    'subgraph {} has no scheduled op'.format(subgraph_id))
            start = min(entry['start'] for entry in subgraph_ops)
            end = max(entry['end'] for entry in subgraph_ops)
            subgraph_entries.append({
                'subgraph_id': subgraph_id,
                'start': start, 'end': end, 'duration': end - start,
            })
        per_core_timeline.append({
            'core_id': core_id,
            'tasks': [{
                'task_id': core_id, 'subgraph_ids': task['subgraph_ids'],
                'start': task_start, 'end': task_end,
                'duration': task_end - task_start,
            }],
            'subgraphs': subgraph_entries,
            'ops': op_entries,
        })

    transfer_timeline = []
    for transfer_id, link in enumerate(cross_links):
        source = key(link['source_core'], link['source_copy_out_id'])
        target = key(link['target_core'], link['target_copy_in_id'])
        item = dict(link)
        item.update({
            'transfer_id': transfer_id,
            'copy_out_end': op_end[source],
            'copy_in_release': op_end[source] + cross_core_copy_delay,
            'copy_in_start': op_start[target],
            'copy_in_end': op_end[target],
        })
        transfer_timeline.append(item)

    result = {
        'scene': 'B',
        'makespan': max(op_end.values(), default=0),
        'num_cores': num_cores,
        'bandwidth_bytes_per_cycle': bandwidth,
        'capacity_bytes': dict(capacity),
        'memory_peak_by_core': {
            core_id: dict(task['step3']['memory_peak'])
            for core_id, task in tasks.items()
        },
        'step3_by_core': {
            core_id: {
                'local_makespan': task['step3']['makespan'],
                'memory_dependency_count': len(
                    task['step3']['memory_dependencies']),
                'pipe_op_counts': {
                    pipe: len(order) for pipe, order in task['pipe_ops'].items()
                },
            }
            for core_id, task in tasks.items()
        },
        'cross_core_copy_delay_cycles': cross_core_copy_delay,
        'cross_task_traffic': cross_task_traffic,
        'data_movement_bytes': data_movement,
        'task_count': num_cores,
        'task_dependencies': cross_links,
        'cross_core_transfers': transfer_timeline,
        'per_core_timeline': per_core_timeline,
    }
    return result


if __name__ == '__main__':
    from contest_io import run_problem_cli
    raise SystemExit(run_problem_cli(2))
