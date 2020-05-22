import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as func


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_input):
        x_input = self.hidden(x_input)
        x_input = func.relu(x_input)
        x_input = self.predict(x_input)
        return x_input


if __name__ == "__main__":
    LR_1 = 0.05
    LR_2 = 0.05
    LR_3 = 0.05
    Alpha_1 = 0.96
    Alpha_2 = 0.96
    Alpha_3 = 0.96

    x_1 = torch.unsqueeze(torch.linspace(0.001, 1.3, 200), 1)
    x_2 = torch.unsqueeze(torch.linspace(1.301, 2.6, 200), 1)
    y_1 = np.log(-x_1 + 1.3001) + 1 * torch.rand(x_1.size()) + 6
    y_2 = - (np.log(x_2 - 1.3001) + 1 * torch.rand(x_2.size())) - 6
    x = torch.cat((x_1, x_2), 0)
    y = torch.cat((y_1, y_2), 0)

    # 相对来说，alpha值越大，拟合程度越高
    net_1 = Net(1, 150, 1)
    optimizer_1 = torch.optim.RMSprop(net_1.parameters(), lr=LR_1, alpha=Alpha_1)
    loss_func_1 = torch.nn.MSELoss()

    net_2 = Net(1, 250, 1)
    optimizer_2 = torch.optim.RMSprop(net_2.parameters(), lr=LR_2, alpha=Alpha_2)
    loss_func_2 = torch.nn.MSELoss()

    net_3 = Net(1, 500, 1)
    optimizer_3 = torch.optim.RMSprop(net_3.parameters(), lr=LR_3, alpha=Alpha_3)
    loss_func_3 = torch.nn.MSELoss()

    plt.figure()
    plt.ion()

    for t in range(10001):
        prediction_1 = net_1(x)
        loss_1 = loss_func_1(prediction_1, y)
        optimizer_1.zero_grad()
        loss_1.backward()
        optimizer_1.step()

        prediction_2 = net_2(x)
        loss_2 = loss_func_2(prediction_2, y)
        optimizer_2.zero_grad()
        loss_2.backward()
        optimizer_2.step()

        prediction_3 = net_3(x)
        loss_3 = loss_func_3(prediction_3, y)
        optimizer_3.zero_grad()
        loss_3.backward()
        optimizer_3.step()

        if t % 100 == 0:
            # print(t, float('%.6f' % loss.data.numpy()))
            plt.cla()
            plt.grid()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction_1.data.numpy(), 'r-.', lw=3)
            plt.plot(x.data.numpy(), prediction_2.data.numpy(), 'b-.', lw=3)
            plt.plot(x.data.numpy(), prediction_3.data.numpy(), 'k-.', lw=3)
            plt.text(0, 0, 'Loss=%.6f' % loss_1.data.numpy(), fontdict={'size': 20, 'color': 'red'})
            plt.text(0, -2, 'Loss=%.6f' % loss_2.data.numpy(), fontdict={'size': 20, 'color': 'blue'})
            plt.text(0, -4, 'Loss=%.6f' % loss_3.data.numpy(), fontdict={'size': 20, 'color': 'black'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
