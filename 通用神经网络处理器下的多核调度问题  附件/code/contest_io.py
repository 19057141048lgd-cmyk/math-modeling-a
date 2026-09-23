"""赛题脚本共用的命令行输入输出。

本文件只处理四件事：读取 JSON、读取固定配置、调用评估函数、写出结果。
算法本身全部留在 ``schedule_step*.py`` 和三个问题评估器中。这样阅读算法
时不需要穿过大量日志与调试代码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from functools import wraps


def cli_errors(function):
    """Report invalid input/scheduling errors without exposing a Python traceback."""
    @wraps(function)
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError, OSError) as error:
            emit_error('[EVALUATION ERROR] {}'.format(error))
            return 1
    return run


def emit(message):
    print(message)


def emit_error(message):
    print(message, file=sys.stderr)


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def _read_json(path):
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f'{path}: duplicate JSON key {key!r}')
            value[key] = item
        return value
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique_object)


def _write_json(path, value):
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def format_data_movement_log(result):
    """返回一行便于检查的数据搬运摘要。"""
    movement = result.get('data_movement_bytes', {})
    cache = result.get('cache_stats', {})
    parts = [
        'scheduled_copy_bytes={}'.format(
            movement.get('scheduled_copy_bytes', 0)),
        'physical_ddr_bytes={}'.format(
            movement.get('physical_ddr_bytes',
                         movement.get('scheduled_copy_bytes', 0))),
    ]
    if cache:
        parts.extend([
            'cache_hits={}/{}'.format(
                cache.get('hits', 0), cache.get('accesses', 0)),
            'cache_hit_bytes={}'.format(cache.get('hit_bytes', 0)),
        ])
    return 'data movement: ' + ', '.join(parts)


def format_multicore_result_log(filename, result):
    """生成稳定、简短的人类可读日志。完整细节始终保存在结果 JSON。"""
    lines = [
        'input_graph: {}'.format(filename),
        'scene: {}'.format(result.get('scene', 'A')),
        'makespan: {}'.format(result['makespan']),
        'num_cores: {}'.format(result['num_cores']),
        format_data_movement_log(result),
    ]
    for core in result.get('per_core_timeline', []):
        ops = core.get('ops', [])
        end = max((op['end'] for op in ops), default=0)
        lines.append('core {}: ops={}, end={}'.format(
            core['core_id'], len(ops), end))
    return '\n'.join(lines) + '\n'


def _trace_json(filename, result):
    """把操作时间线转换为 Perfetto/Chrome 可以打开的 Trace JSON。"""
    events = []
    pipe_order = {'PIPE_MTE2': 0, 'PIPE_MTE3': 1, 'PIPE_M': 2, 'PIPE_V': 3}
    # Use one Perfetto process track per core.  Without explicit metadata the
    # UI labels numeric tids as generic ``Thread`` tracks and loses the core /
    # pipe hierarchy.
    pid_base = 1000
    for core in result.get('per_core_timeline', []):
        core_id = core['core_id']
        pid = pid_base + core_id
        events.append({'name': 'process_name', 'ph': 'M', 'pid': pid, 'tid': 0,
                       'args': {'name': 'Core {}'.format(core_id)}})
        for pipe, index in pipe_order.items():
            events.append({'name': 'thread_name', 'ph': 'M', 'pid': pid,
                           'tid': index, 'args': {'name': pipe}})
    for core in result.get('per_core_timeline', []):
        core_id = core['core_id']
        pid = pid_base + core_id
        # Evaluators already publish task/subgraph spans. Keep them as their
        # own category so the core process can be collapsed/expanded while
        # the four pipe tracks remain visible underneath it.
        subgraphs = core.get('subgraphs', [])
        if not subgraphs:
            grouped = {}
            for op in core.get('ops', []):
                sg = op.get('subgraph_id', op.get('task_id'))
                if sg is None:
                    continue
                item = grouped.setdefault(sg, {'subgraph_id': sg,
                                               'task_id': op.get('task_id'),
                                               'start': op['start'],
                                               'end': op['end']})
                item['start'] = min(item['start'], op['start'])
                item['end'] = max(item['end'], op['end'])
            subgraphs = list(grouped.values())
        for span in subgraphs:
            start, end = int(span['start']), int(span['end'])
            if end < start:
                continue
            sg = span.get('subgraph_id', span.get('task_id'))
            events.append({
                'name': 'Subgraph {}'.format(sg), 'cat': 'SUBGRAPH', 'ph': 'X',
                'ts': start, 'dur': end - start, 'pid': pid, 'tid': 100,
                'args': {'core_id': core_id, 'task_id': span.get('task_id'),
                         'subgraph_id': sg, 'start': start, 'end': end},
            })
        for op in core.get('ops', []):
            pipe = op['pipe']
            events.append({
                'name': '{} #{}'.format(op['op'], op['op_id']),
                'cat': pipe,
                'ph': 'X',
                'ts': op['start'],
                'dur': op['duration'],
                'pid': pid,
                'tid': pipe_order.get(pipe, 9),
                'args': {
                    'core_id': core_id,
                    'task_id': op.get('task_id'),
                    'subgraph_id': op.get('subgraph_id'),
                    'op_id': op['op_id'],
                    'pipe': pipe,
                },
            })
    return json.dumps({
        'traceEvents': events,
        'displayTimeUnit': 'cycles',
        'otherData': {
            'input_graph': filename,
            'scene': result.get('scene'),
            'makespan': result['makespan'],
            'track_layout': 'one Perfetto process group per core; pipe threads are named PIPE_*',
            'subgraph_spans': 'X events named Subgraph <id> carry start/end in args',
        },
    }, ensure_ascii=False, indent=2)


def format_scene_a_trace_json(filename, result):
    return _trace_json(filename, result)


def format_scene_b_trace_json(filename, result):
    return _trace_json(filename, result)


def _common_paths(args, problem_name):
    graph_path = Path(args.graph)
    stem = graph_path.with_suffix('')
    plan = Path(args.plan) if args.plan else Path(str(stem) + '_multicore_res.json')
    config = Path(args.config) if args.config else graph_path.parent / 'config.txt'
    output = Path(args.output) if args.output else Path(
        str(stem) + '_{}_res.json'.format(problem_name))
    trace = Path(args.trace_output) if args.trace_output else Path(
        str(stem) + '_{}_trace.json'.format(problem_name))
    log = Path(args.log_output) if args.log_output else Path(
        str(stem) + '_{}_log.txt'.format(problem_name))
    return graph_path, plan, config, output, trace, log


@cli_errors
def run_problem_cli(problem, argv=None):
    """三个问题共用一个命令行流程，避免维护三份重复代码。"""
    parser = argparse.ArgumentParser(
        description='评估问题 {} 的多核执行时间'.format(problem))
    parser.add_argument('graph', help='原始计算图 JSON')
    parser.add_argument('plan', nargs='?', help='多核方案 JSON；默认取同名文件')
    parser.add_argument('--config', help='固定评估配置；默认取图所在目录/config.txt')
    parser.add_argument('-o', '--output', help='结果 JSON 路径')
    parser.add_argument('--trace-output', help='Perfetto Trace JSON 路径')
    parser.add_argument('--log-output', help='简短文本日志路径')
    args = parser.parse_args(argv)
    if args.config and not Path(args.config).is_file():
        raise FileNotFoundError(f'configuration file not found: {args.config}')

    name = 'problem_{}'.format(problem)
    graph_path, plan_path, config_path, output, trace, log = _common_paths(
        args, name)
    # 容量/带宽/Pipe 槽位统一来自 config.txt；没有配置文件就不再评估，
    # 避免用代码内置数值给出错误成绩。
    if str(config_path) != args.config and not config_path.is_file():
        raise FileNotFoundError(
            'configuration file not found: {} (defaults to <graph dir>/config.txt; '
            'pass --config to choose another file)'.format(config_path))
    graph, plan = _read_json(graph_path), _read_json(plan_path)

    from evaluation_validation import read_evaluation_config
    settings = read_evaluation_config(str(config_path))
    capacity = settings['capacity']
    bandwidth = settings['bandwidth']

    if problem == 1:
        from multicore_cut_evaluate_problem_1 import evaluate_scene_a, read_scene_a_config
        scene = read_scene_a_config(str(config_path))
        result = evaluate_scene_a(
            graph, plan, bandwidth=bandwidth, capacity=capacity,
            cross_core_wait=scene['task_cross_core_wait_cycles'],
            same_core_wait=scene['task_same_core_wait_cycles'])
        trace_text = format_scene_a_trace_json(graph_path.name, result)
    elif problem == 2:
        from multicore_cut_evaluate_problem_2 import evaluate_scene_b, read_scene_b_config
        scene = read_scene_b_config(str(config_path))
        result = evaluate_scene_b(
            graph, plan, bandwidth=bandwidth, capacity=capacity,
            cross_core_copy_delay=scene['cross_core_copy_delay_cycles'])
        trace_text = format_scene_b_trace_json(graph_path.name, result)
    elif problem == 3:
        from multicore_cut_evaluate_problem_3 import (
            evaluate_problem_3, read_cache_config, read_scene_b_config)
        scene = read_scene_b_config(str(config_path))
        cache = read_cache_config(str(config_path))
        result = evaluate_problem_3(
            graph, plan, bandwidth=bandwidth, capacity=capacity,
            cross_core_copy_delay=scene['cross_core_copy_delay_cycles'],
            **cache)
        trace_text = format_scene_b_trace_json(graph_path.name, result)
    else:
        raise ValueError('problem must be 1, 2, or 3')

    result['input_graph'] = graph_path.name
    result['input_plan'] = plan_path.name
    _write_json(output, result)
    write_text(trace, trace_text + '\n')
    write_text(log, format_multicore_result_log(graph_path.name, result))
    emit(format_data_movement_log(result))
    emit('OK: problem {} makespan={}; result={}'.format(
        problem, result['makespan'], output))
    return 0


@cli_errors
def run_multicore_stub_cli(argv=None):
    """生成一份格式合法的随机示例方案；它不是参考优化算法。"""
    from stub_multicore_cut_and_schedule import (
        DEFAULT_MAX_SUBGRAPH_SIZE, DEFAULT_MIN_SUBGRAPH_SIZE,
        DEFAULT_NUM_CORES, DEFAULT_SEED, generate_multicore_plan)
    parser = argparse.ArgumentParser(description='生成多核方案格式示例')
    parser.add_argument('graph')
    parser.add_argument('-n', '--num-cores', type=int, default=DEFAULT_NUM_CORES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--min-subgraph-size', type=int,
                        default=DEFAULT_MIN_SUBGRAPH_SIZE)
    parser.add_argument('--max-subgraph-size', type=int,
                        default=DEFAULT_MAX_SUBGRAPH_SIZE)
    parser.add_argument('-o', '--output')
    args = parser.parse_args(argv)
    graph_path = Path(args.graph)
    plan = generate_multicore_plan(
        _read_json(graph_path), num_cores=args.num_cores, seed=args.seed,
        min_subgraph_size=args.min_subgraph_size,
        max_subgraph_size=args.max_subgraph_size)
    output = Path(args.output) if args.output else Path(
        str(graph_path.with_suffix('')) + '_multicore_res.json')
    _write_json(output, plan)
    emit('OK: {} ops -> {} subgraphs on {} cores; output={}'.format(
        len(plan['node_to_subgraph']),
        len(set(plan['node_to_subgraph'].values())),
        len(plan['core_schedules']), output))
    return 0
