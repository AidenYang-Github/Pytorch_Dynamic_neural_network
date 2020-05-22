# Variable变量

import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1, 2], [3, 4]])
variable = Variable(tensor, requires_grad=True)

# print(tensor)
# print(variable)

t_out = torch.mean(tensor*tensor)   # x^2
v_out = torch.mean(variable*variable)

# print(t_out)
# print(v_out)

v_out.backward()
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = (1/4)*2*variable = variable/2

# print(variable.grad)
print(variable)
print(variable.data)
print(variable.data.numpy())
