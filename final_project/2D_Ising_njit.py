import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import time
from tqdm import tqdm
from matplotlib.colors import ListedColormap

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

#@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8, f8, f8, unicode_type)", nogil=True)
# above line doesn't work when the metropolis function returns the final spin config
# numba doesn't support type casting 2D arrays (?)
# but using the above line instead of the one below, speeds up the code in the first function call
# there is no difference in the consecutive function calls
@njit(nogil=True)
def metropolis(spin_config, iterations, J, h, beta, energy, mode='toroid'):
    N = len(spin_config)
    spin_config = spin_config.copy()
    tot_spins = np.zeros(iterations)
    tot_energy = np.zeros(iterations)
    for i in range(0, iterations):
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
        tot_spins[i] = np.sum(spin_config)
        tot_energy[i] = energy

    # return the arrays of total spins and energies over the iterations and the final spin config 
    return tot_spins, tot_energy, spin_config


# init values
N = 50
J = 1
h = 0
beta = 1
iter = 100000
mode = 'toroid' # choose between 'toroid' and 'flat'

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
    energy_n = energy_toroid(lattice_n, J, h)
    energy_p = energy_toroid(lattice_p, J, h)
if mode == 'flat':
    energy_n = energy_flat(lattice_n, J, h)
    energy_p = energy_flat(lattice_p, J, h)

# metropolis algo
start = time.time()
spins_n, energies_n, dummy = metropolis(lattice_n, iter, J, h, beta, energy_n, mode)
spins_p, energies_p, dummy = metropolis(lattice_p, iter, J, h, beta, energy_p, mode)
print('Runtime:', np.round(time.time()-start, 2), 's')

# plot the results
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins_n/N**2)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Average Spin $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energies_n)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy', y=1.07, size=18)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(spins_p/N**2)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Average Spin $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(energies_p)
ax.set_xlabel('Algorithm Time Steps')
ax.set_ylabel(r'Energy $E/J$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Evolution of Average Spin and Energy', y=1.07, size=18)
plt.show()

# phase transition, starting with the equilibrium state for beta = 1 (thermalisation)
lattice_p = dummy
beta = np.linspace(0, 1, 50)
netmag = np.zeros(len(beta))
for i in tqdm(range(len(netmag))):
    totspin, totenergy, dummy = metropolis(lattice_p, iter, J, h, beta[i], energy_p, mode)
    netmag[i] = totspin[-1]/(N*N)

#plot the results
plt.plot(beta, netmag, '.')
plt.show()