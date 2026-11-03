import torch
from torch import nn
from d2l import torch as d2l

def init_normal(module):
    if type(module) == nn.Linear:
        nn.init.normal_(module.weight, mean=0, std=0.01)
        nn.init.zeros_(module.bias)

def init_constant(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 1)
        nn.init.zeros_(module.bias)

net = nn.Sequential(nn.LazyLinear(8), nn.ReLU(), nn.LazyLinear(1))
X = torch.rand(size=(2,4))
print(net(X).shape)

def init_xavier(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)

def init_42(module):
    if type(module) == nn.Linear:
        nn.init.constant_(module.weight, 42)

def my_init(module):
    if type(module) == nn.Linear:
        print("Init", *[(name, param.shape) for name, param in module .named_parameters()][0])
        nn.init.uniform_(module.weight, -10, 10)
        module.weight.data *= module.weight.data.abs() >= 5

@d2l.add_to_class(d2l.Module)
def apply_init(self, inputs, init=None):
    self.forward(*inputs)

    if init is not None:
        self.net.apply(init)    

net.apply(init_normal)
print(net[0].weight.data[0], net[0].bias.data[0])
net.apply(init_constant)
print(net[0].weight.data[0], net[0].bias.data[0])
net[0].apply(init_xavier)
net[2].apply(init_42)
print(net[0].weight.data[0])
print(net[2].weight.data)
net.apply(my_init)
print(net[0].weight[2:])

# Manually modify single parameters
net[0].weight.data[:] += 1
net[0].weight.data[0,0] = 42
print(net[0].weight.data[0])

net = nn.Sequential(nn.LazyLinear(256), nn.ReLU(), nn.LazyLinear(10))
print(net[0].weight) # Parameters are un-initialized "<UninitializedParameter>"

X = torch.rand(2, 20)
net(X)
print(net[0].weight.shape)