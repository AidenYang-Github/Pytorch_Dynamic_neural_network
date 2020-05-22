import torch
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(1)  # reproducible

# === hyper parameters ===
LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# === 建立数据集 ===
y_txt = []
filename = 'HXD1D-0323#4#9#WD.txt'
file = open(filename)
text = file.readlines()
for t in text:
    t_value = t[15:21]
    t_value = float(t_value)
    if t_value > 0:
        y_txt.append(t_value)

x_txt = np.linspace(0, len(y_txt) - 1, len(y_txt))
x = torch.from_numpy(x_txt)
y = torch.from_numpy(np.array(y_txt))
x = torch.unsqueeze(x, dim=1).float()
y = torch.unsqueeze(y, dim=1).float()
# x = torch.unsqueeze(torch.linspace(0, len(y_txt) - 1, len(y_txt)), dim=1)     # x的两种写法
# x_1 = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# y_1 = x_1.pow(2) + 0.1 * torch.normal(torch.zeros(*x_1.size()))
# plot dataset
# plt.scatter(x.numpy(), y.numpy())
# plt.show()

# === data loader ===
# torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)


# === default network ===
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 200)
        self.predict = torch.nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


# === different nets ===
net_SGD = Net()
net_Momentum = Net()
net_RMSprop = Net()
net_Adam = Net()
nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

# === different optimizers ===
opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

loss_func = torch.nn.MSELoss()
losses_his = [[], [], [], []]  # 记录training时不同神经网络的loss
# """
if __name__ == '__main__':
    for epoch in range(EPOCH):
        print('Epoch:', epoch)
        for step, (b_x, b_y) in enumerate(loader):
            # 对每个优化器，优化属于它的神经网络
            for net, opt, l_his in zip(nets, optimizers, losses_his):
                output = net(b_x)
                loss = loss_func(output, b_y)

                opt.zero_grad()
                loss.backward()
                opt.step()
                l_his.append(loss.data.numpy())

    labels = ['SGD', 'Momentum', 'RMSPROP', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 0.2)
    plt.grid()
    plt.show()

    plt.ion()
    plt.show()
    for t in range(10001):
        prediction_SGD = net_SGD(x)  # 传输给net训练数据x，输出预测值
        prediction_Momentum = net_Momentum(x)
        prediction_RMSprop = net_RMSprop(x)
        prediction_Adam = net_Adam(x)

        loss_SGD = loss_func(prediction_SGD, y)  # 计算两者的误差
        loss_Momentum = loss_func(prediction_Momentum, y)  # 计算两者的误差
        loss_RMSprop = loss_func(prediction_RMSprop, y)  # 计算两者的误差
        loss_Adam = loss_func(prediction_Adam, y)  # 计算两者的误差

        opt_SGD.zero_grad()  # 清空上一步的残余更新参数值
        loss_SGD.backward()  # 误差反向传播，计算参数更新值
        opt_SGD.step()  # 将参数更新值施加到net的parameters上

        opt_Momentum.zero_grad()  # 清空上一步的残余更新参数值
        loss_Momentum.backward()  # 误差反向传播，计算参数更新值
        opt_Momentum.step()  # 将参数更新值施加到net的parameters上

        opt_RMSprop.zero_grad()  # 清空上一步的残余更新参数值
        loss_RMSprop.backward()  # 误差反向传播，计算参数更新值
        opt_RMSprop.step()  # 将参数更新值施加到net的parameters上

        opt_Adam.zero_grad()  # 清空上一步的残余更新参数值
        loss_Adam.backward()  # 误差反向传播，计算参数更新值
        opt_Adam.step()  # 将参数更新值施加到net的parameters上
        # ==== 可视化训练过程 ====
        if t % 100 == 0:
            plt.cla()
            # print(t, 'loss_SGD=', float('%.6f' % loss_SGD.data.numpy()))
            # print(t, 'loss_Momentum=', float('%.6f' % loss_Momentum.data.numpy()))
            # print(t, 'loss_RMSprop=', float('%.6f' % loss_RMSprop.data.numpy()))
            # print(t, 'loss_Adam=', float('%.6f' % loss_Adam.data.numpy()))
            plt.grid()
            plt.plot(x.data.numpy(), y.data.numpy(), '.')
            plt.plot(x.data.numpy(), prediction_SGD.data.numpy(), 'r-.', lw=3)
            plt.plot(x.data.numpy(), prediction_Momentum.data.numpy(), 'y-.', lw=3)
            # plt.plot(x.data.numpy(), prediction_RMSprop.data.numpy(), 'g-.', lw=3)
            plt.plot(x.data.numpy(), prediction_Adam.data.numpy(), 'm-.', lw=3)
            plt.text(150, 1220, 'Loss_SGD=%.6f' % loss_SGD.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.text(150, 1218, 'loss_Momentum=%.6f' % loss_Momentum.data.numpy(),
                     fontdict={'size': 20, 'color': 'yellow'})
            # plt.text(150, 1216, 'loss_RMSprop=%.6f' % loss_RMSprop.data.numpy(),
            #          fontdict={'size': 20, 'color': 'green'})
            plt.text(150, 1214, 'loss_Adam=%.6f' % loss_Adam.data.numpy(), fontdict={'size': 20, 'color': 'magenta'})

            plt.pause(0.01)

        plt.ioff()
        plt.show()
# """
