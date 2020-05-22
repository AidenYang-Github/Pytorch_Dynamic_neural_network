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
    x_1 = torch.unsqueeze(torch.linspace(0.001, 1.3, 200), 1)
    x_2 = torch.unsqueeze(torch.linspace(1.301, 2.6, 200), 1)
    y_1 = np.log(-x_1 + 1.3001) + 1 * torch.rand(x_1.size()) + 6
    y_2 = - (np.log(x_2 - 1.3001) + 1 * torch.rand(x_2.size())) - 6
    x = torch.cat((x_1, x_2), 0)
    y = torch.cat((y_1, y_2), 0)

    net = Net(1, 500, 1)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.7)
    optimizer = torch.optim.RMSprop(net.parameters(), lr=0.05, alpha=0.9)
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
