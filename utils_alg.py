import numpy as np 
import torch
from torch.utils.data import WeightedRandomSampler
import matplotlib.pyplot as plt



# Stochastic Imbalanced Partition

# hard_mask = False: This set a distribution (probability of being sampled) over the data classes to mimic imbalance datasets.
# The distribuiton is given by a power law.
# Across different epochs different data would be sample
#
# hard_mask = True: Same as before but this time the dataset is partitioned completely. 
# This "weights" mask set a priori which data element is present (samplable) or not, 
# while the former instead gives only a distribution over the data classes.
# So this time the same data will be sampled across epochs.

def SIP(dataset, targets, batch_size, alpha=0.03, beta=0.2, hard_mask=True, plot_dist=False):

    if alpha == 0 and beta == 0:
        weights = torch.ones(len(targets))
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader, weights

    l_targ = len(targets)
    vals = torch.arange(l_targ)
    np.random.shuffle(vals.numpy())
    weights = torch.clip(torch.exp(vals*-alpha*l_targ) + beta, min=0, max=1)
    samples_weights = weights[dataset.targets]

    if hard_mask:
        for i,w in enumerate(samples_weights):
            if np.random.random() <= w:
                samples_weights[i] = 1
            else:
                samples_weights[i] = 0

    sampler = WeightedRandomSampler(
        weights=samples_weights,
        num_samples=len(samples_weights),
        replacement=True)
    
    if plot_dist:
        plt.plot(weights)
        plt.show()

    data_loader = torch.utils.data.DataLoader(dataset, drop_last=True, sampler = sampler, batch_size=batch_size)
    return data_loader, weights



# Test sampling
def test_sampling(pool,emulated_devices,sample_prob):
    sampled_val = []
    k = 0
    for i in range(emulated_devices):
        val = sample_prob()
        print(val, val<pool/emulated_devices )
        sampled_val.append(val)
        if val<pool/emulated_devices :
            k += 1
    print("Sampled Devices: ",k)
    plt.title("Sampled Values")
    plt.plot(np.sort(sampled_val))
    plt.show()
    return k