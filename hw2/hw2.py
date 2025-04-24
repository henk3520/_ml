import random
import numpy as np

# 城市坐標
citys = [
    (0,3), (0,0), (0,2), (0,1),
    (1,0), (1,3), (2,0), (2,3),
    (3,0), (3,3), (3,1), (3,2)
]

# 計算兩點之間的歐幾里得距離
def distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# 計算路徑總長度
def pathLength(path, citys):
    dist = 0
    plen = len(path)
    for i in range(plen):
        dist += distance(citys[path[i]], citys[path[(i + 1) % plen]])
    return dist

# 高度函數（負路徑長度，因為爬山演算法最小化）
def height(path, citys):
    return pathLength(path, citys)

# 生成鄰居路徑（2-opt 交換）
def neighbor(path):
    new_path = path.copy()
    n = len(path)
    i, j = sorted(random.sample(range(n), 2))  # 隨機選擇兩個不同索引
    new_path[i:j + 1] = reversed(new_path[i:j + 1])  # 反轉子路徑
    return new_path

# 爬山演算法（最小化）
def hillClimbing(x, height, neighbor, citys, max_fail=10000):
    fail = 0
    current_height = height(x, citys)
    while True:
        nx = neighbor(x)
        nx_height = height(nx, citys)
        if nx_height < current_height:  # 尋找更短的路徑
            x = nx
            current_height = nx_height
            fail = 0
        else:
            fail += 1
            if fail > max_fail:
                return x, current_height

# 主程序
def solve_tsp(seed=42):
    random.seed(seed)
    
    # 生成隨機初始路徑
    n_cities = len(citys)
    path = list(range(n_cities))
    random.shuffle(path)  # 隨機排列城市
    
    # 運行爬山演算法
    best_path, best_distance = hillClimbing(path, height, neighbor, citys)
    
    return best_path, best_distance

# 運行並輸出結果
best_path, best_distance = solve_tsp()
print("城市數:", len(citys))
print("最佳路徑:", best_path + [best_path[0]])  # 顯示閉合路徑
print("總距離:", best_distance)
print("城市坐標:")
for i, coord in enumerate(citys):
    print(f"城市 {i}: {coord}")
