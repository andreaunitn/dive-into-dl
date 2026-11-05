import torch
from torch import nn
from d2l import torch as d2l

# Define a helper function to calculate convolutions. It Initializes the conv layer weights and performs corresponding dimensionality elevations and reductions on the input and output
def comp_conv2d(conv2d, X):
    # (1, 1) indicated batch size and num channels are the same
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)

    # Strip the first two dimensions: sample & channels
    return Y.reshape(Y.shape[2:])

def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform cross-correlation with input X. All results are stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]

    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))

    # Matrix mul in FC layer
    Y = torch.matmul(K, X)

    return Y.reshape((c_o, h, w))

# 1 row and column is padded on either side
conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1)
X = torch.rand(size=(8,8))
print(comp_conv2d(conv2d, X).shape)

# Use a convolution kernel with h = 5 and w = 3. Padding is (2,1)
conv2d = nn.LazyConv2d(1, kernel_size=(5,3), padding=(2,1))
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.LazyConv2d(1, kernel_size=3, padding=1, stride=2)
print(comp_conv2d(conv2d, X).shape)

conv2d = nn.LazyConv2d(1, kernel_size=(3,5), padding=(0,1), stride=(3,4))
print(comp_conv2d(conv2d, X).shape)

X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                  [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(corr2d_multi_in(X, K))

K = torch.stack((K, K+1, K+2), 0)
print(K.shape)
print(X)
print(corr2d_multi_in_out(X, K))

X = torch.normal(0, 1, (3,3,3))
K = torch.normal(0, 1, (2,3,1,1))
Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)
assert float(torch.abs(Y1 - Y2).sum()) < 1e-6