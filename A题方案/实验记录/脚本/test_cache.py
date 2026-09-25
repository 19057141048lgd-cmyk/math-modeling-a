import os
import sys

os.chdir(r'C:\Users\richd\Desktop\数学建模\A题(1)\A题方案')
sys.path.insert(0, '.')
from solver.sched import CacheModel

c = CacheModel(1048576)
c.insert('A', 350000, 0)
c.insert('B', 350000, 10)
before = c.present('A', 50)
c.insert('C', 350000, 100)
print('50 时刻查询 A：插入 C 之前=%s，之后=%s（应均为 True）' % (before, c.present('A', 50)))
print('150 时刻查询 A（A+B+C=1.05MB > 1MB，应为 False）:', c.present('A', 150))
print('150 时刻查询 B（B+C 应为 True）:', c.present('B', 150))
c.insert('D', 350000, 5)   # 提交顺序晚、时刻早
print('50 时刻查询 A（A+D+B=1.05MB，应为 False）:', c.present('A', 50))
print('50 时刻查询 D（D+B，应为 True）:', c.present('D', 50))
print('3 时刻查询 D（尚未插入，应为 False）:', c.present('D', 3))
c.insert('A', 350000, 200)
print('250 时刻查询 A（重新插入后，应为 True）:', c.present('A', 250))
