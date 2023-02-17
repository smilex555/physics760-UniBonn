import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.ndimage import convolve, generate_binary_structure
import time

def energy_flat(spin_config, J, h):
    kern = generate_binary_structure(2, 1)
    kern[1][1] = False
    nrg = -J*0.5*np.sum(spin_config * convolve(spin_config, kern, mode='constant', cval=0)) - h*np.sum(spin_config)
    return nrg

def energy_toroid(spin_config, J, h):
    left = np.roll(spin_config, 1, 1)
    right = np.roll(spin_config, -1, 1)
    up = np.roll(spin_config, 1, 0)
    down = np.roll(spin_config, -1, 0)
    energy = -J*0.5*np.sum(spin_config*(left + right + up +down)) - h*np.sum(spin_config)
    return energy

@njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8, f8, f8, unicode_type)", nogil=True)
def metropolis(spin_config, iterations, J, h, beta, energy, mode='toroid'):
    N = len(spin_config)
    spin_config = spin_config.copy()
    tot_spins = np.zeros(iterations)
    tot_energy = np.zeros(iterations)
    for i in range(0, iterations):
        # random point on array and flip spin
        x = np.random.randint(0,N)
        y = np.random.randint(0,N)
        spin_i = spin_config[x,y] #initial spin
        spin_f = spin_i*-1 #proposed spin flip
        
        if mode == 'toroid':
            # compute energy change: toroidal geometry
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

        if mode == 'flat':
            # compute energy change: flat geometry
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
        
        # Change state with designated probabilities
        dE = E_f - E_i
        if (dE>0)*(np.random.random() < np.exp(-beta*dE)):
            spin_config[x,y] = spin_f
            energy += dE
        elif dE<=0:
            spin_config[x,y] = spin_f
            energy += dE
            
        tot_spins[i] = np.sum(spin_config)
        tot_energy[i] = energy
            
    return tot_spins, tot_energy

N = 50
J = 0.2
h = 0
beta = 1
mode = 'toroid' # choose between 'toroid' and 'flat'
    
init_random = np.random.random((N, N))
lattice_n = np.zeros((N, N))
lattice_n[init_random >= 0.75] = 1
lattice_n[init_random < 0.75] = -1

init_random = np.random.random((N, N))
lattice_p = np.zeros((N, N))
lattice_p[init_random <= 0.75] = 1
lattice_p[init_random > 0.75] = -1

plt.imshow(lattice_n)
plt.show()
plt.imshow(lattice_p)
plt.show()

if mode == 'toroid':
    energy_n = energy_toroid(lattice_n, J, h)
    energy_p = energy_toroid(lattice_p, J, h)
if mode == 'flat':
    energy_n = energy_flat(lattice_n, J, h)
    energy_p = energy_flat(lattice_p, J, h)

start = time.time()
spins_n, energies_n = metropolis(lattice_n, 1000000, J, h, beta, energy_n, mode)
spins_p, energies_p = metropolis(lattice_p, 1000000, J, h, beta, energy_p, mode)
print('Runtime:', time.time()-start)

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
fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.2', y=1.07, size=18)
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
fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.2', y=1.07, size=18)
plt.show()