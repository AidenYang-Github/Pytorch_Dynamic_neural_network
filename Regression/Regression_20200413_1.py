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
    x = torch.unsqueeze(torch.linspace(0.001, 1.3, 400), 1)
    # y = np.tan(x) + 0.2 * torch.rand(x.size())
    # y = x ** 2 + 4 * np.tan(x) + 0.2 * torch.rand(x.size())
    # y = x ** 3 - np.tan(x) + 0.2 * torch.rand(x.size())
    # y = np.tanh(x) + 0.2 * torch.rand(x.size())
    # y = -np.tanh(x) + 0.2 * torch.rand(x.size())
    # y = (x + 0.5) ** 3 + (x - 0.5) ** 3 - np.tan(x + 0.5) - np.tan(x - 0.5) + 0.2 * torch.rand(x.size())
    # y = -1 * (np.log(x) + 1 * torch.rand(x.size()))
    # y = np.log(-x + 1.3001) + 1 * torch.rand(x.size()) - (np.log(x) + 1 * torch.rand(x.size()))
    y = np.log(-x + 1.3001) - (np.log(x) + 1 * torch.rand(x.size()))
    # y = np.log(-x + 1.3001) + 1 * torch.rand(x.size())
    # y = - (np.log(x) + 1 * torch.rand(x.size()))

    net = Net(1, 500, 1)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.7)
    loss_func = torch.nn.MSELoss()

    plt.figure()
    plt.ion()

    for t in range(10001):
        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t % 100 == 0:
            print(t, float('%.6f' % loss.data.numpy()))
            plt.cla()
            plt.grid()
            plt.scatter(x.data.numpy(), y.data.numpy())
            plt.plot(x.data.numpy(), prediction.data.numpy(), 'k-.', lw=3)
            plt.text(0, -4, 'Loss=%.6f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'black'})
            plt.pause(0.1)
    plt.ioff()
    plt.show()
