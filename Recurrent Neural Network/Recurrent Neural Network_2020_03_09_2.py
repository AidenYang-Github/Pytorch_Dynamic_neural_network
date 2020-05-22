import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 训练数据
# Hyper Parameters
TIME_STEP = 10  # rnn time step/image height
INPUT_SIZE = 1  # rnn input size / image width
LR = 0.02  # learning rate
DOWNLOAD_MNIST = False  # set to True if haven't download the data


# RNN模型
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # 有几层RNN layers
            batch_first=True,  # input&output 会是以batch size为第一维度的特征集 e.g.(batc, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):  # 因为hidden state 是连续的，所以我们要一直传递这一个state
        # x(batch,time_state,input_size)
        # h_state(n_layers,batch,hidden_size)
        # r_out(batch,time_step,output_size)
        r_out, h_state = self.rnn(x, h_state)  # h_state也要作为RNN的一个输入

        outs = []
        for time_step in range(r_out.size(1)):
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

    """
    def forward(self, x, h_state):
        r_out, h_state = self.rnn(x, h_state)
        r_out = r_out.view(-1, 32)
        outs = self.out(r_out)
        return outs.view(-1, 32, TIME_STEP), h_state
    """


rnn = RNN()
print(rnn)
"""
RNN(
  (rnn): RNN(1, 32, batch_first=True)
  (out): Linear(in_features=32, out_features=1, bias=True)
)"""

# 训练
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimizer all rnn parameters
loss_func = nn.MSELoss()

h_state = None  # 要使用初始hidden state,可以设成None

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time steps
    # sin 预测cos
    steps = np.linspace(start, end, 10, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape(batch,time_step,input_size
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    prediction, h_state = rnn(x, h_state)  # rnn对于每个step的prediction,还有最后一个step的h_state
    # 下面很重要！！
    h_state = h_state.data  # 要把h_state重新包装一下才能放入下一个iteration，不然会报错

    loss = loss_func(prediction, y)  # cross entropy loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
