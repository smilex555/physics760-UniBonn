import numpy as np
import random
import matplotlib.pyplot as plt

# Blocking / Binning the markov chain data
def blocking(mc_data, bin_width):
    if len(mc_data) % bin_width == 0:
        N_bin = len(mc_data) // bin_width
    else:
        N_bin = len(mc_data) // bin_width + 1  
    mean_block = np.zeros(N_bin)
    #idx = np.arange(0,N_bin,1)
    for j in range(N_bin):
        i = bin_width*j
        if j >= N_bin - 1:
            mean_block[j] = np.mean(mc_data[i:])
        else:
            mean_block[j] = np.mean(mc_data[i:i+bin_width])
    
    return mean_block    



# bootstrap routine
def bootstrap(data,n_bs):
    mean_bs = np.zeros(n_bs)

    for i in range(n_bs):
        N_config = len(data)
        
        # choose new random configuration of the markov chain
        bs_idx = np.random.randint(0,N_config, size=N_config)
        
        # new data configuration with the random drawn indeces 
        bs_data = data[bs_idx]
        
        # mean of the new data config
        mean_bs[i] = np.mean(bs_data)
        
    return np.mean(mean_bs), np.std(mean_bs)


###
num_bs = 200 # number of bootstrap samples
bs_bin = [1,2,4,8,10,15,20,30] # used bin width
bs_mean = np.zeros(len(bs_bin))
bs_std = np.zeros(len(bs_bin))
observ = np.random.normal(0,2,1000) 

for b_i,bin in enumerate(bs_bin):
    mc_block = blocking(observ,bin)
    bs_mean[b_i], bs_std[b_i] = bootstrap(mc_block, num_bs)

print('Mean and Std (Different Bin Widths:', bs_mean, bs_std)