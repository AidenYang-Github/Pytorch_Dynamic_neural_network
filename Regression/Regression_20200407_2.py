import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as func


# == 搭建神经网络模型 ==
class Net(torch.nn.Module):  # 继承 torch 的 Module/
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        # 定义每层用什么样的形式
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x_input):  # 这同时也是 Module 中的 forward 功能
        # 正向传播输入值, 神经网络分析出输出值
        x_input = self.hidden(x_input)
        x_input = func.relu(x_input)  # 激励函数(隐藏层的线性值)
        x_input = self.predict(x_input)  # 输出值
        return x_input


if __name__ == "__main__":
    x = torch.unsqueeze(torch.linspace(-1.5, 1.5, 200), dim=1)  # x data (tensor), shape=(100, 1)
    y_1 = x.pow(2) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    y_2 = x.pow(3) + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    # y_3 = x.pow(5) + 0.5 + 0.2 * torch.rand(x.size())  # noisy y data (tensor), shape=(100, 1)
    y_3 = np.sin(x) + 0.2 * torch.rand(x.size())

    # net_1 = Net(n_feature=1, n_hidden=500, n_output=1)
    # optimizer_1 = torch.optim.SGD(net_1.parameters(), lr=0.05)  # 传入 net 的所有参数, 学习率
    # loss_func_1 = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)
    # net_2 = Net(n_feature=1, n_hidden=500, n_output=1)
    # optimizer_2 = torch.optim.SGD(net_2.parameters(), lr=0.05)  # 传入 net 的所有参数, 学习率
    # loss_func_2 = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)
    net_3 = Net(n_feature=1, n_hidden=500, n_output=1)
    # optimizer_3 = torch.optim.SGD(net_3.parameters(), lr=0.05)  # 传入 net 的所有参数, 学习率
    optimizer_3 = torch.optim.SGD(net_3.parameters(), lr=0.05, momentum=0.8)  # 传入 net 的所有参数, 学习率
    loss_func_3 = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)

    print('momentum=', optimizer_3.defaults['momentum'])
    plt.figure()
    plt.ion()  # 可视化训练过程

    for t in range(10001):
        # prediction_1 = net_1(x)  # 喂给 net 训练数据 x, 输出预测值
        # loss_1 = loss_func_1(prediction_1, y_1)  # 计算两者的误差
        # optimizer_1.zero_grad()  # 清空上一步的残余更新参数值
        # loss_1.backward()  # 误差反向传播, 计算参数更新值
        # optimizer_1.step()  # 将参数更新值施加到 net 的 parameters 上
        #
        # prediction_2 = net_2(x)  # 喂给 net 训练数据 x, 输出预测值
        # loss_2 = loss_func_2(prediction_2, y_2)  # 计算两者的误差
        # optimizer_2.zero_grad()  # 清空上一步的残余更新参数值
        # loss_2.backward()  # 误差反向传播, 计算参数更新值
        # optimizer_2.step()  # 将参数更新值施加到 net 的 parameters 上

        prediction_3 = net_3(x)  # 喂给 net 训练数据 x, 输出预测值
        loss_3 = loss_func_3(prediction_3, y_3)  # 计算两者的误差
        optimizer_3.zero_grad()  # 清空上一步的残余更新参数值
        loss_3.backward()  # 误差反向传播, 计算参数更新值
        optimizer_3.step()  # 将参数更新值施加到 net 的 parameters 上

        if t % 100 == 0:
            # plot and show learning process
            # print(t, float('%0.6f' % loss_1.data.numpy()), float('%0.6f' % loss_2.data.numpy()),
            #       float('%0.6f' % loss_3.data.numpy()))
            print(t, float('%0.6f' % loss_3.data.numpy()))
            plt.cla()
            plt.grid()
            # plt.scatter(x.data.numpy(), y_1.data.numpy())
            # plt.scatter(x.data.numpy(), y_2.data.numpy())
            plt.scatter(x.data.numpy(), y_3.data.numpy())
            # plt.plot(x.data.numpy(), prediction_1.data.numpy(), 'r-.', lw=3)
            # plt.plot(x.data.numpy(), prediction_2.data.numpy(), 'b-.', lw=3)
            plt.plot(x.data.numpy(), prediction_3.data.numpy(), 'k-.', lw=3)
            # plt.text(0, -0.1, 'Loss=%.6f' % loss_1.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            # plt.text(0, -0.25, 'Loss=%.6f' % loss_2.data.numpy(), fontdict={'size': 20, 'color': 'blue'})
            plt.text(0, -0.4, 'Loss=%.6f' % loss_3.data.numpy(), fontdict={'size': 20, 'color': 'black'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
