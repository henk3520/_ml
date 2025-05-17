import torch
import numpy as np
import matplotlib.pyplot as plt

# 數據
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

# 將數據轉換為 PyTorch 張量
x_tensor = torch.tensor(x, requires_grad=False).reshape(-1, 1)
y_tensor = torch.tensor(y, requires_grad=False).reshape(-1, 1)

# 初始化參數（截距 b 和斜率 w），需要梯度
b = torch.tensor(0.0, requires_grad=True)
w = torch.tensor(0.0, requires_grad=True)

# 學習率和訓練次數
learning_rate = 0.01
epochs = 3000

# 訓練過程
for epoch in range(epochs):
    # 前向傳播
    y_pred = w * x_tensor + b
    
    # 計算均方誤差 (MSE) 損失
    loss = torch.mean((y_pred - y_tensor) ** 2)
    
    # 反向傳播
    loss.backward()
    
    # 更新參數（使用梯度下降）
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        
        # 清除梯度
        w.grad.zero_()
        b.grad.zero_()

# 預測值，直接使用 NumPy 計算
y_predicted = w.item() * x + b.item()

# 繪製圖表
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()

# 打印結果
print('w=', w.item())
print('b=', b.item())
print('y_predicted=', y_predicted)
