import torch
import numpy as np
import matplotlib.pyplot as plt

# x = torch.unsqueeze(torch.linspace(0.001, 1.3, 200), 1)
x_2 = torch.unsqueeze(torch.linspace(0.501, 2.6, 400), 1)
# y = np.log(-x_2 + 1.3001) + 1 * torch.rand(x_2.size())
y = - (np.log(x_2 - 0.5) + 1 * torch.rand(x_2.size())) + 6

plt.figure()
plt.plot(x_2, y, '.')
plt.grid()
plt.show()
