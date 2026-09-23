"""官方附件定位与 config.txt 解析。"""
import os
from dataclasses import dataclass

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(HERE)
PROBLEM_DIR = os.path.dirname(SOLUTION_DIR)


def official_dir():
    """在 A题 目录下自动寻找官方附件（包含 code/multicore_cut_evaluate_problem_1.py 的目录）。"""
    env = os.environ.get('A_OFFICIAL_DIR')
    if env:
        return env
    for name in sorted(os.listdir(PROBLEM_DIR)):
        path = os.path.join(PROBLEM_DIR, name)
        if os.path.isfile(os.path.join(path, 'code', 'multicore_cut_evaluate_problem_1.py')):
            return path
    raise FileNotFoundError('未找到官方附件目录，请设置环境变量 A_OFFICIAL_DIR')


def data_dir():
    return os.path.join(official_dir(), 'data')


def code_dir():
    return os.path.join(official_dir(), 'code')


def case_path(case):
    return os.path.join(data_dir(), case + '.json')


@dataclass
class HwConfig:
    l1: int = 524288
    ub: int = 131072
    bandwidth: float = 60.0
    cross_wait_a: float = 1000.0
    same_wait_a: float = 100.0
    cross_delay_b: float = 500.0
    cache_capacity: int = 1048576
    cache_bandwidth: float = 250.0


def load_config(path=None):
    path = path or os.path.join(data_dir(), 'config.txt')
    values = {}
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if not line or line.startswith('['):
                continue
            key, value = line.split()
            values[key] = float(value)
    return HwConfig(
        l1=int(values.get('L1', 524288)),
        ub=int(values.get('UB', 131072)),
        bandwidth=values.get('bandwidth', 60.0),
        cross_wait_a=values.get('task_cross_core_wait_cycles', 1000.0),
        same_wait_a=values.get('task_same_core_wait_cycles', 100.0),
        cross_delay_b=values.get('cross_core_copy_delay_cycles', 500.0),
        cache_capacity=int(values.get('cache_capacity_bytes', 1048576)),
        cache_bandwidth=values.get('cache_bandwidth_bytes_per_cycle', 250.0),
    )
