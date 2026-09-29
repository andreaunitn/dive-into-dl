import torch
from d2l import torch as d2l

class SoftmaxRegressionScratch(d2l.Classifier):
    def __init__(self, num_inputs, num_outputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()

        self.W = torch.normal(0, sigma, size=(num_inputs, num_outputs), requires_grad=True)
        self.b = torch.zeros(num_outputs, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

@d2l.add_to_class(SoftmaxRegressionScratch) #@save
def forward(self, X):
    X = X.reshape((-1, self.W.shape[0])) # Flatten the 28x28 image into a single 784 vector.
    return softmax(torch.matmul(X, self.W) + self.b)

@d2l.add_to_class(SoftmaxRegressionScratch) #@save
def loss(self, y_hat, y):
    return cross_entropy(y_hat, y)

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdims=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[list(range(len(y_hat))), y]).mean()

X = torch.rand((2,5))
X_prob = softmax(X)
print(X_prob)
print(X_prob.sum(1))

y = torch.tensor([0,2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
print(y_hat[[0,1], y])
print(cross_entropy(y_hat, y))

data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegressionScratch(num_inputs=784, num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)

X, y = next(iter(data.val_dataloader()))
preds = model(X).argmax(axis=1)
print(preds.shape)

wrong = preds.type(y.dtype) != y
X, y, preds = X[wrong], y[wrong], preds[wrong]
labels = [a+"\n"+b for a,b in zip(data.text_labels(y), data.text_labels(preds))]
data.visualize([X, y], labels=labels)