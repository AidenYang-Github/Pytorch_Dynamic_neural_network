import torch
import torch.nn.functional as func

x = torch.linspace(-5, 5, 200)  # x data (tensor), shape=(100,1)
x_np = x.data.numpy()
y_relu = func.relu(x).data.numpy()
y_sigmoid_1 = func.sigmoid(x).data.numpy()  # 这行是视频中的代码
y_sigmoid_2 = torch.sigmoid(x).data.numpy()  # 这行是修改后的代码
