import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve, generate_binary_structure
from numba import njit
import time
from tqdm import tqdm

# function to calculate energy in a flat geometry, i.e.,
# points on the boundary do not have points from the opposite edges as neighbours
def energy_flat(spin_config, J, h):
    ''' This function takes in a spin config, values of J and h
    and returns the total energy of the spin config in flat geometry'''
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    nrg = -J*0.5*np.sum(spin_config * convolve(spin_config, kern, mode='constant', cval=0)) - h*np.sum(spin_config)
    return nrg

# function to calculate energy in a toroidal geometry, i.e.,
# points on the boundary have points from the opposite edges as neighbours
def energy_toroid(spin_config, J, h):
    ''' This function takes in a spin config, values of J and h
    and returns the total energy of the spin config in toroidal geometry'''
    left = np.roll(spin_config, 1, 1)
    right = np.roll(spin_config, -1, 1)
    up = np.roll(spin_config, 1, 0)
    down = np.roll(spin_config, -1, 0)
    energy = -J*0.5*np.sum(spin_config*(left + right + up +down)) - h*np.sum(spin_config)
    return energy

# the first line where the type is asserted explicitly during numba call
# doesn't work when the metropolis function returns the final spin config
# numba doesn't support type asserting 2D arrays (???)
# but using the second line instead of the first, speeds up the code during the first function call
# there is no difference in the consecutive function calls

#@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8, f8, f8, unicode_type)", nogil=True)
@njit(nogil=True)
def metropolis(spin_config, iterations, burnin, J, h, beta, energy, mode='toroid'):
    '''Metropolis algorithm for a 2D Ising Model.
    Takes in an initial spin config, no. of iterations and burn-in steps,
    J, h, beta, energy of the initial spin config and default energy mode as toroidal.
    Returns net magnetisation, energy vs. algo time steps
    net magnetisation, energy without burnin vs algo time steps and final spin config'''
    N = len(spin_config)
    spin_config = spin_config.copy()
    tot_spins_noburnin = np.zeros(iterations + burnin)
    tot_energy_noburnin = np.zeros(iterations + burnin)
    tot_spins = np.zeros(iterations)
    tot_energy = np.zeros(iterations)
    for i in range(0, iterations + burnin):
        # step 1: pick a random point flip spins
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_config[x,y]
        spin_f = -1*spin_i
        
        # step 2.1: energy change for toroidal geometry
        if mode == 'toroid':
            E_i = 0
            E_f = 0
            E_i += -J*spin_i*spin_config[(x-1)%N, y]
            E_i += -J*spin_i*spin_config[(x+1)%N, y]
            E_i += -J*spin_i*spin_config[x, (y-1)%N]
            E_i += -J*spin_i*spin_config[x, (y+1)%N]
            E_i += -h*spin_i
            E_f += -J*spin_f*spin_config[(x-1)%N, y]
            E_f += -J*spin_f*spin_config[(x+1)%N, y]
            E_f += -J*spin_f*spin_config[x, (y-1)%N]
            E_f += -J*spin_f*spin_config[x, (y+1)%N]
            E_f += -h*spin_f

        # step 2.2: energy change for flat geometry
        if mode == 'flat':
            E_i = 0
            E_f = 0
            if x>0:
                E_i += -J*spin_i*spin_config[x-1,y]
                E_f += -J*spin_f*spin_config[x-1,y]
            if x<N-1:
                E_i += -J*spin_i*spin_config[x+1,y]
                E_f += -J*spin_f*spin_config[x+1,y]
            if y>0:
                E_i += -J*spin_i*spin_config[x,y-1]
                E_f += -J*spin_f*spin_config[x,y-1]
            if y<N-1:
                E_i += -J*spin_i*spin_config[x,y+1]
                E_f += -J*spin_f*spin_config[x,y+1]
            E_i += -h*spin_i
            E_f += -h*spin_f
        
        # step 3: determine whether to accept the change
        dE = E_f - E_i
        if (dE>0)*(np.random.random() < np.exp(-beta*dE)):
            spin_config[x,y] = spin_f
            energy += dE
        elif dE<=0:
            spin_config[x,y] = spin_f
            energy += dE
        
        # save the values
        tot_spins_noburnin[i] = np.sum(spin_config)
        tot_energy_noburnin[i] = energy
        if i >= burnin:
            tot_spins[i - burnin] = np.sum(spin_config)
            tot_energy[i - burnin] = energy

    # return the arrays of total spins and energies over the iterations and the final spin config 
    return tot_spins, tot_energy, tot_spins_noburnin, tot_energy_noburnin, spin_config


# init values
N = 50
j = 1.
h = 0.
beta = 1.
iter = 100000
burn = 30000
mode = 'toroid' # choose between 'toroid' and 'flat'

# test if everything's working as expected
# -0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0-

# create a mostly negative initial config
init_random = np.random.random((N, N))
lattice_n = np.zeros((N, N))
lattice_n[init_random >= .8] = 1
lattice_n[init_random < .8] = -1

# create a mostly positive initial config
init_random = np.random.random((N, N))
lattice_p = np.zeros((N, N))
lattice_p[init_random <= .8] = 1
lattice_p[init_random > .8] = -1

# use black for -1 spin, white for +1 spin
cmap = ListedColormap(["black", "white"])
plt.imshow(lattice_n, cmap=cmap)
plt.show()
plt.imshow(lattice_p, cmap=cmap)
plt.show()

# calculate the total energy of the two spin config
if mode == 'toroid':
    energy_n = energy_toroid(lattice_n, j, h)
    energy_p = energy_toroid(lattice_p, j, h)
if mode == 'flat':
    energy_n = energy_flat(lattice_n, j, h)
    energy_p = energy_flat(lattice_p, j, h)

# metropolis algo
start = time.time()
spins_n, energies_n, spinsnob_n, energiesnob_n, equi_n = metropolis(lattice_n, iter, burn, j, h, beta, energy_n, mode)
spins_p, energies_p, spinsnob_p, energiesnob_p, equi_p = metropolis(lattice_p, iter, burn, j, h, beta, energy_p, mode)
print('Runtime:', np.round(time.time()-start, 2), 's')

# plot the results

# without burn-in
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spinsnob_n/(N*N))
ax.set_xlabel('Algorithm Iteration')
ax.set_ylabel(r'Net magnetisation $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energiesnob_n)
ax.set_xlabel('Algorithm Iteration')
ax.set_ylabel(r'Energy $E$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Net Magnetisation and Energy', y = 1, size=12)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spinsnob_p/(N*N))
ax.set_xlabel('Algorithm Iteration')
ax.set_ylabel(r'Net Magnetisation $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energiesnob_p)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy', y=1, size=12)
plt.show()

# with burn-in
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins_n/(N*N))
ax.set_xlabel('Algorithm Iteration')
ax.set_ylabel(r'Net magnetisation $\bar{m}$')
ax.set_ylim([-1., 1.])
ax.grid()
ax = axes[1]
ax.plot(energies_n)
ax.set_xlabel('Algorithm Iteration')
ax.set_ylabel(r'Energy $E$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Net Magnetisation and Energy', y = 1, size=12)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins_p/(N*N))
ax.set_xlabel('Algorithm Iteration')
ax.set_ylabel(r'Net Magnetisation $\bar{m}$')
ax.set_ylim([-1., 1.])
ax.grid()
ax = axes[1]
ax.plot(energies_p)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy', y=1, size=12)
plt.show()

# -0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0-

# phase transition
# -0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0-

beta = np.linspace(0, 1, 80)

netmag_p = np.zeros(len(beta))
for i in tqdm(range(len(netmag_p))):
    totspin, totenergy, totspinnob, totenergynob, dummy = metropolis(lattice_p, iter, burn, j, h, beta[i], energy_p, mode)
    netmag_p[i] = np.average(totspin)/(N*N)

netmag_n = np.zeros(len(beta))
for i in tqdm(range(len(netmag_n))):
    totspin, totenergy, totspinnob, totenergynob, dummy = metropolis(lattice_p, iter, burn, j, h, beta[i], energy_p, mode)
    netmag_n[i] = np.average(totspin)/(N*N)

#plot the results
plt.plot(beta, netmag_p, '.')
plt.show()
plt.plot(beta, netmag_n, '.')
plt.show()
# -0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0--0-0-0-0-0-