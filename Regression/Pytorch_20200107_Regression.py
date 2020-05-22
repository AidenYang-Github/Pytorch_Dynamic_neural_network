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
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承__init__功能
        # 定义每层用什么样的形式
        self.linear1 = torch.nn.Linear(n_feature, n_hidden)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n_hidden, n_output)
        # 设置优化器，选择误差计算方式
        self.opt = torch.optim.SGD(self.parameters(), lr=0.005)  # 随机梯度下降优化器
        self.criterion = torch.nn.MSELoss()  # 均方差

    def forward(self, x_input):  # 这同时也是Module中的forward功能
        # 正向传播输入值，神经网络分析出输出值
        y_output = self.linear1(x_input)  # 激励函数(隐藏层的线性值)
        y_output = self.relu(y_output)
        y_output = self.linear2(y_output)
        return y_output


# ===== 训练网络 =====
net = Net(1, 1000, 1)
# print(model)  # 查看net的结构
plt.ion()
plt.show()
for t in range(10001):
    prediction = net(x)  # 传输给net训练数据x，输出预测值

    loss = net.criterion(prediction, y)  # 计算两者的误差

    net.opt.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播，计算参数更新值
    net.opt.step()  # 将参数更新值施加到net的parameters上
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
