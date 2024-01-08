import numpy as np

# print(np.random.randn(8) * 0)
# import numpy as np

# 假设 phi_k 和 w_k 是 NumPy 数组
phi_k = np.array([1, 2, 3])  # 特征向量 phi_k
w_k = np.array([0.5, 0.2, 0.8])  # 权重向量 w_k

# 计算 phi_k^T * w_k
phi_w = np.dot(phi_k, w_k)

print(phi_w)