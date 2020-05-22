import torch
import numpy as np

"""
np_data = np.arange(6).reshape((2, 3))
torch_data = torch.from_numpy(np_data)
tensor2array = torch_data.numpy()
print('\nnumpy', np_data, '\ntorch', torch_data, '\ntensor2array', tensor2array)
"""
"""
# abs/sin/mean
data = [-1, -2, 1, 2]
tensor = torch.FloatTensor(data)

print('\nabs', '\nnumpy:', np.abs(data), '\ntorch:', torch.abs(tensor))
print('\nsin', '\nnumpy:', np.sin(data), '\ntorch:', torch.sin(tensor))
print('\nmean', '\nnumpy:', np.mean(data), '\ntorch:', torch.mean(tensor))
"""
# 矩阵相乘
data = [[1, 2], [3, 4]]
tensor = torch.FloatTensor(data)  # 32-bit float point
print('\nnumpy:', np.matmul(data, data),
      '\ntorch:', torch.mm(tensor, tensor))

data1 = np.array(data)
data1.dot(data1)
tensor1 = torch.FloatTensor(data1.flatten())
print('\nnumpy:', data1.dot(data1),
      '\ntorch:', tensor1.dot(tensor1))
