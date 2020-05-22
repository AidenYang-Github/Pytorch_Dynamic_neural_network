import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as Data

# === 为了对比各种优化器的效果，建立伪数据 ===
torch.manual_seed(1)

LR = 0.01
BATCH_SIZE = 32
EPOCH = 12

# fake dataset
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

# plot dataset
# plt.scatter(x.numpy(), y.numpy())
# plt.grid()
# plt.show()

# 使用data loader
torch_dataset = Data.TensorDataset(x, y)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# === 每个优化器优化一个神经网络，但这个神经网络都来自同一个Net形式 ===
# 默认的network形式
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)  # hidden layer
        self.predict = torch.nn.Linear(20, 1)  # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


if __name__ == '__main__':
    # 为每个优化器创建一个net
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    # === 优化器 Optimizer ===
    # 接下来在创建不同的优化器，用来训练不同的网络。
    # 并创建一个loss_func用来计算误差。
    # 常用几种常见的优化器：SGD，Momentum,RMSprop,Adam

    # different optimizer
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]  # 记录training时不同神经网络的loss

    # === 训练/作图 ===

    for epoch in range(EPOCH):
        print('Epoch:', epoch)
        for step, (b_x, b_y) in enumerate(loader):

            # 对每个优化器，优化属于各自的神经网络
            for net, opt, I_his in zip(nets, optimizers, losses_his):
                output = net(b_x)  # get output for every net
                loss = loss_func(output, b_y)  # compute loss for every net
                opt.zero_grad()  # clear gradients for next train
                loss.backward()  # backpropagation, compute gradient
                opt.step()  # apply gradients
                I_his.append(loss.data.numpy())  # loss recorder
