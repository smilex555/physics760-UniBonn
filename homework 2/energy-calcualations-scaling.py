import numpy as np
import timeit

#below is the function that we use to calculate the total energy of a given state
def energy_ising(J, spin_config, h):
    left = np.roll(spin_config, 1, 1)
    right = np.roll(spin_config, -1, 1)
    up = np.roll(spin_config, 1, 0)
    down = np.roll(spin_config, -1, 0)
    energy = (1/2)*-J*np.sum(spin_config*(left + right + up +down))- h*np.sum(spin_config)
    return energy

#generate 20 arbitrary configurations of sizes 1, 2, ... and time the energy calculations
for k in range(1, 21):
    config = np.zeros((k, k))
    for i in range(0, k):
        for j in range(0, k):
            if np.random.rand() < 0.5: config[i, j] = -1
            else: config[i, j] = 1
    print(f'N={k}:', timeit.timeit(lambda: energy_ising(1, config, 0)), 'microsec') #lambda adds significant overhead to the times
    #use the below line for times closer to the actual times
    #%timeit energy_ising(1, config, 0) #use this line on JupyterHub or iPython console