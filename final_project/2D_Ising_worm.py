import numpy as np
import random

def worm_update(lattice, J, beta):
    N = lattice.shape[0]
    M = lattice.shape[1]
    worm_head = [np.random.randint(N), np.random.randint(M)]
    worm_tail = [worm_head[0], worm_head[1]]
    #worm_tail = [worm_head[0] - direction[0], worm_head[1] - direction[1]]
    worm_sites = [worm_head, worm_tail]
    
    # Check if the worm can grow in the current direction
    while True:
        direction = random.choice([[0,1], [0,-1], [1,0], [-1,0]])
        next_site = [ (worm_head[0] + direction[0])%N, (worm_head[1] + direction[1])%M ]
        #if next_site[0] < 0 or next_site[0] >= N or next_site[1] < 0 or next_site[1] >= M:
        #    break
        if random.uniform(0,1) < np.tanh(J*beta):
            worm_head = next_site
            worm_sites.append(worm_head)
        if lattice[next_site[0], next_site[1]] == lattice[worm_head[0], worm_head[1]]:
            break
        worm_head = next_site
        worm_sites.append(worm_head)
    
    # Try to flip the worm
#    energy_diff = 2 * lattice[worm_head[0], worm_head[1]] * np.sum([lattice[site[0], site[1]] for site in worm_sites])
#    if energy_diff <= 0 or random.uniform(0,1) < np.exp(-beta * energy_diff):
    for site in worm_sites:
        lattice[site[0], site[1]] *= -1
    
    return lattice

def simulate_ising_model(N, M, J, beta, num_steps):
    lattice = np.random.choice([-1, 1], size=(N, M))
    for step in range(num_steps):
        lattice = worm_update(lattice, J, beta)
    return lattice
