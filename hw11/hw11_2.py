from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import numpy as np

# 生成隨機數據
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)

# 訓練 K-Means 模型
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# 獲取分群結果
labels = kmeans.labels_
print("Cluster labels:", labels)
