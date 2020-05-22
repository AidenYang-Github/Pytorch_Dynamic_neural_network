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


rnn = RNN()
print(rnn)
"""
RNN(
  (rnn): RNN(1, 32, batch_first=True)
  (out): Linear(in_features=32, out_features=1, bias=True)
)"""