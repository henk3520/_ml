import random
import numpy as np

# 目標函數 f(x, y, z)
def height(x):
    return x[0]**2 + x[1]**2 + x[2]**2 - 2*x[0] - 4*x[1] - 6*x[2] + 8

# 鄰居生成函數
def neighbor(x, step_size=0.1):
    # 在當前點附近隨機擾動
    nx = x.copy()
    nx[0] += random.uniform(-step_size, step_size)
    nx[1] += random.uniform(-step_size, step_size)
    nx[2] += random.uniform(-step_size, step_size)
    return nx

# 爬山演算法（最小化）
def hillClimbing(x, height, neighbor, max_fail=10000):
    fail = 0
    while True:
        nx = neighbor(x)
        if height(nx) < height(x):  # 尋找更小的函數值
            x = nx
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return x

# 運行爬山演算法
random.seed(42)  # 設置隨機種子以確保可重現
initial_point = [random.uniform(-10, 10) for _ in range(3)]  # 隨機初始點
print("初始點:", initial_point)
result = hillClimbing(initial_point, height, neighbor)
print("找到的最低點:", result)
print("函數值:", height(result))
