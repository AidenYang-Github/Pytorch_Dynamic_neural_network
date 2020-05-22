# Regression:回归
import torch
from torch.autograd import Variable
import torch.nn.functional as F  # 激励函数在functional中
import matplotlib.pyplot as plt

# ====== 建立数据集 =====
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

x, y = Variable(x), Variable(y)


# plt.scatter(x.data.numpy(), y.data.numpy())  # scatter：绘制散点图
# plt.show()


# ===== 建立神经网络 =====
class Net(torch.nn.Module):  # 继承torch的Module
    def __init__(self, n_features, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_features, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):  # 这同时也是Module中的forward功能
        # 正向传播输入值，神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.predict(x)  # 输出值
        return x


net = Net(1, 1000, 1)
print(net)  # 查看net的结构

plt.ion()
plt.show()

# ===== 训练网络 =====
# optimizer（优化器）是训练的工具
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)  # 传入net的所有参数，学习率
# optimizer = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))  # 传入net的所有参数，学习率
loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式（均方差）

for t in range(10001):
    prediction = net(x)  # 传输给net训练数据x，输出预测值

    loss = loss_func(prediction, y)  # 计算两者的误差

    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    optimizer.step()  # 将参数更新值施加到net的parameters上
    # ==== 可视化训练过程 ====
    if t % 100 == 0:
        plt.cla()
        print(t, loss.data)
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-.', lw=3)
        plt.text(-0.25, 1, 'Loss=%.6f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)

    plt.ioff()
    plt.show()
