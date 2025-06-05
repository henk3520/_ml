import torch
import torch.nn as nn
import torch.optim as optim

# 輸入：0-9 的 one-hot 編碼
X = torch.eye(10)

# 輸出：七段顯示器的真值表
Y = torch.tensor([
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1],  # 9
], dtype=torch.float32)

# 定義 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(10, 3),  # 隱藏層節點數改為 3 (參考你之前的圖片)
            nn.ReLU(),
            nn.Linear(3, 7),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化模型、損失函數和優化器
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 使用 SGD 以模擬簡單梯度下降

# 訓練
num_epochs = 5000  # 增加迭代次數以確保收斂
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, Y)
    loss.backward()
    optimizer.step()
    
    # 每 500 次打印一次損失
    if (epoch + 1) % 500 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")

# 預測並打印結果
pred = model(X).round()
print("\n預測結果：")
for i in range(10):
    pred_list = pred[i].int().tolist()
    print(f"數字 {i}: {pred_list} (a-g)")

# 檢查預測是否正確
correct = (pred == Y).all(dim=1)
for i in range(10):
    print(f"數字 {i} 預測{'正確' if correct[i] else '錯誤'}")
