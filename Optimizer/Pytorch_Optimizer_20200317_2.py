import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
import torch.utils.data as torch_data

torch.manual_seed(1)
# hyper parameters
LR = 0.01   # 学习率
BATCH_SIZE = 32
EPOCH = 12

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(3) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = torch_data.TensorDataset(x, y)
loader = torch_data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)


# default network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(1, 20)
        self.predict = torch.nn.Linear(20, 1)

    def forward(self, x_input):
        x_input = func.relu(self.hidden(x_input))
        x_input = self.predict(x_input)
        return x_input


if __name__ == '__main__':
    # different nets
    net_SGD = Net()
    net_Momentum = Net()
    net_RMSprop = Net()
    net_Adam = Net()
    nets = [net_SGD, net_Momentum, net_RMSprop, net_Adam]

    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.8)
    opt_RMSprop = torch.optim.RMSprop(net_RMSprop.parameters(), lr=LR, alpha=0.9)
    opt_Adam = torch.optim.Adam(net_Adam.parameters(), lr=LR, betas=(0.9, 0.99))
    optimizers = [opt_SGD, opt_Momentum, opt_RMSprop, opt_Adam]

    loss_func = torch.nn.MSELoss()
    losses_his = [[], [], [], []]

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

    labels = ['SGD', 'Momentum', 'RMSprop', 'Adam']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.ylim(0, 0.2)
    plt.grid()
    plt.show()
