import torch
from torch import nn
from d2l import torch as d2l

class BNLeNetScatch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5),
            BatchNorm(6, num_dims=4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            BatchNorm(16, num_dims=4),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            BatchNorm(120, num_dims=2),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            BatchNorm(84, num_dims=2),
            nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()

        self.net = nn.Sequential(
            nn.LazyConv2d(6, kernel_size=5),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LazyConv2d(16, kernel_size=5),
            nn.LazyBatchNorm2d(),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(120),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),
            nn.LazyLinear(84),
            nn.LazyBatchNorm1d(),
            nn.Sigmoid(),
            nn.LazyLinear(num_classes)
        )

class BatchNorm(nn.Module):
    # num_features: number of outputs for a FC layer or number of output channels for conv layer
    # num_dims: = 2 for FC, 4 for conv
    def __init__(self, num_features, num_dims):
        super().__init__()

        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)

        self.gamma = nn.Parameter(torch.ones(shape)) # Scale
        self.beta = nn.Parameter(torch.zeros(shape)) # Shift

        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If X is not in main memory, copy mean and variance in the same device as X
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)

        Y, self.moving_mean, self.moving_var = batch_norm(X, self.gamma, self.beta, self.moving_mean, self.moving_var, eps=1e-5, momentum=0.1)
        return Y

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use is_grad_enabled to determine whether we are in training mode
    if not torch.is_grad_enabled():
        # In inference use mean and variance obtained by moving average (over all dataset)
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)

    else:
        assert len(X.shape) in (2,4) # Distinguish between linear layer (2) and conv layer (4)
        if len(X.shape) == 2: # Linear layer
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)

        else: # Conv layer
            mean = X.mean(dim=(0,2,3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0,2,3), keepdim=True)

        X_hat = (X - mean) / torch.sqrt(var + eps)

        # Update mean and variance using moving average
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean
        moving_var = (1.0 - momentum) * moving_var + momentum * var

    Y = gamma * X_hat + beta # Scale and shift
    return Y, moving_mean.data, moving_var.data

trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScatch(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
d2l.plt.show()

print(model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1)))

trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
d2l.plt.show()