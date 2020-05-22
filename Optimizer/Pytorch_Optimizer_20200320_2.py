import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
import torch.utils.data as torch_data


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


torch.manual_seed(1)
LR = 0.01
BATCH_SIZE = 10
EPOCH = 10

x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

torch_dataset = torch_data.TensorDataset(x, y)
loader = torch_data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

if __name__ == '__main':
    net_SGD = Net(1, 20, 1)
    nets = [net_SGD]
    opt_SGD = torch.optim.SGD(net_SGD.parameters(), lr=LR)
    opts = [opt_SGD]
    loss_func = torch.nn.MSELoss()
    losses_his = [[]]

    for epoch in range(EPOCH):
        print('Epoch:', epoch)
        for step, (b_x, b_y) in enumerate(loader):
            for net, opt, l_his in zip(nets, opts, losses_his):
                output = net(b_x)
                loss = loss_func(output, b_y)

                opt.zero_grad()
                loss.backward()
                opt.step()

    labels = ['SGD']
    for i, l_his in enumerate(losses_his):
        plt.plot(l_his, label=labels[i])
    plt.legend(loc='best')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.grid()
    plt.show()
