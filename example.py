import matplotlib
import random
import torch
from torch.distributions.multinomial import Multinomial
from d2l import torch as d2l

num_tosses = 100
heads = sum(random.random() > 0.5 for _ in range(num_tosses))
tails = num_tosses - heads
print(f'Number of heads: {heads}, Number of tails: {tails}')

fair_probs = torch.tensor([0.5, 0.5])
Multinomial_probs = Multinomial(num_tosses, fair_probs).sample() / 100
print(f'Multinomial sample: {Multinomial_probs}')

counts = Multinomial(10000, fair_probs).sample()
print(f'{counts / 10000}')

counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label='P(coin=heads)')
d2l.plt.plot(estimates[:, 1], label='P(coin=tails)')
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated Probability')
d2l.plt.legend()