import torch
from torch import nn

net = nn.Sequential(
    nn.LazyLinear(8),
    nn.ReLU(),
    nn.LazyLinear(1)
)

X = torch.rand(size=(2,4))
print(net(X).shape)
print(net[2].state_dict()) # Parameters of the second fully connected layer
print(type(net[2].bias))
print(net[2].bias.data) # Access the value of the bias from the second fully connected layer
print(net[2].weight.grad == None)

# Accessing parameters of all layers
print([(name, param.shape) for name, param in net.named_parameters()])

shared = nn.LazyLinear(8)
net = nn.Sequential(
    nn.LazyLinear(8),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared,
    nn.ReLU(),
    nn.LazyLinear(1)
)

print(net(X))
print(net[2].weight.data[0] == net[4].weight.data[0]) # Check whether the parameters are the same
net[2].weight.data[0] = 100
print(net[2].weight.data[0] == net[4].weight.data[0])