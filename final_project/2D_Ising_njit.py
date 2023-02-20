#!/usr/bin/python
# libraries
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from numba import njit
import time
from tqdm import tqdm

# function to calculate energy in a toroidal geometry, i.e.,
# lattice points on the boundary have lattice points from the opposite edges as neighbours
def energy_toroid(spin_config, J, h):
    '''Calculate total energy of a given spin config in flat geomtery, i.e.,
    lattice points on the edges don't have lattice points on the opposite edge as neighbours.
    Args:
        spin_config (2D numpy.ndarray): Spin config for which the energy is to be calculated
        J (float): Interaction parameter of the system
        h (float): Magnitude of the external magnetic field
    Returns:
        energy (float): Energy of the given spin config
    '''
    left = np.roll(spin_config, 1, 1)
    right = np.roll(spin_config, -1, 1)
    up = np.roll(spin_config, 1, 0)
    down = np.roll(spin_config, -1, 0)
    energy = -J*0.5*np.sum(spin_config*(left + right + up +down)) - h*np.sum(spin_config)
    return energy
# function to calculate energy in flat geometry is now depreciated and can be found in ./obsolete.py/

# the first line where the type is asserted explicitly during numba call,
# doesn't work when the metropolis function returns the final spin config
# numba doesn't support type asserting 2D arrays (?)
# but using the second line instead of the first, speeds up the code during the first function call
# there is no difference in the consecutive function calls

#@njit("UniTuple(f8[:], 2)(f8[:,:], i8, i8, f8, f8, f8, f8)", nogil=True)
@njit(nogil=True)
def metropolis(spin_config, iterations, burnin, J, h, beta, energy):
    '''Metropolis algorithm for a 2D Ising Model.
    Args:
        spin_config (2D numpy.ndarray): Initial spin configuration
        iterations (int): Total number of Metropolis iterations
        burnin (int): Total number of burn-in iterations
        J (float): Interaction parameter of the system
        h (float): Magnitude of the external magnetic field
        beta (float): Inverse temperature
        energy (float): Energy of the initial spin configuration
    Returns:
        tot_spins (1D numpy.ndarray): Array of net magnetisation at every iteration after burn-in
        tot_energy (1D numpy.ndarray): Array of energy of the spin config at every iteration after burn-in
        tot_spins_noburnin (1D numpy.ndarray): Array of net magnetisation at every iteration before burn-in
        tot_energy_noburnin (1D numpy.ndarray): Array of energy of the spin config at every iteration before burn-in
    '''
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
        
        # step 2: energy change for toroidal geometry
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
    return tot_spins, tot_energy, tot_spins_noburnin, tot_energy_noburnin

def spin_autocorr_time(spins):
    """
    Calculate the spin autocorrelation time for an Ising model from a series of net spin values.
    Args:
        spins (1D numpy.ndarray): 1D NumPy array of net spin values
    Returns:
        tau (float): Spin autocorrelation time
    """
    # Compute the mean and variance of the spin series
    m = np.mean(spins)
    v = np.var(spins)

    # Compute the autocorrelation function of the spin series
    acf = np.correlate(spins - m, spins - m, mode='full')
    acf = acf[len(acf)//2:] / (v * (len(spins) - 1))

    # Find the first index where the autocorrelation function is less than exp(-1)
    idx = np.where(acf < np.exp(-1))[0][0]

    # Compute the spin autocorrelation time
    tau = 2 * np.sum(acf[:idx])

    return tau

# init values
N = 50
j = 1.
h = 0.
beta = 1.
iter = 100000
burn = 30000

# test if everything's working as expected
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

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
energy_n = energy_toroid(lattice_n, j, h)
energy_p = energy_toroid(lattice_p, j, h)

# metropolis algo
start = time.time()
spins_n, energies_n, spinsnob_n, energiesnob_n = metropolis(lattice_n, iter, burn, j, h, beta, energy_n)
spins_p, energies_p, spinsnob_p, energiesnob_p = metropolis(lattice_p, iter, burn, j, h, beta, energy_p)
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

# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

# phase transition
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

beta_arr = np.linspace(0, 1, 50)

netmag_p = np.zeros(len(beta_arr))
for i in tqdm(range(len(netmag_p))):
    totspin, totenergy, totspinnob, totenergynob = metropolis(lattice_p, iter, burn, j, h, beta_arr[i], energy_p)
    netmag_p[i] = np.average(totspin)/(N*N)

netmag_n = np.zeros(len(beta_arr))
for i in tqdm(range(len(netmag_n))):
    totspin, totenergy, totspinnob, totenergynob = metropolis(lattice_p, iter, burn, j, h, beta_arr[i], energy_p)
    netmag_n[i] = np.average(totspin)/(N*N)

#plot the results
fig, axes = plt.subplots(1, 2, figsize=(12,4))
ax = axes[0]
ax.plot(beta_arr, netmag_p, '.')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'Net Magnetisation $\bar{m}$')
ax.grid()
ax = axes[1]
ax.plot(beta_arr, netmag_n, '.')
ax.set_xlabel(r'$\beta$')
ax.set_ylabel(r'Net Magnetisation $\bar{m}$')
ax.grid()
fig.tight_layout()
fig.suptitle(r'Net Magnetisation vs. $\beta$', y=1, size=12)
plt.show()
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

# energy autocorrelation
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

n_array = np.arange(5, 41, 2)

autocorrtime = np.zeros(len(n_array))
for i in tqdm(range(len(n_array))):
    init_random = np.random.random((n_array[i], n_array[i]))
    lattice = np.zeros((n_array[i], n_array[i]))
    lattice[init_random <= .8] = 1
    lattice[init_random > .8] = -1
    energy_t = energy_toroid(lattice, j, h)
    totspin, totenergy, totspinnob, totenergynob = metropolis(lattice, iter, burn, j, h, beta, energy_t)
    autocorrtime[i] = spin_autocorr_time(totspin)

# fit to verify tau-lattice size scaling
def fitf(x, m, c):
    return m*x + c

popt, pcov = curve_fit(fitf, np.log(n_array), np.log(autocorrtime))
print('Fit slope:', np.round(popt[0], 2))

# plot the results
xrange = np.linspace(np.min(np.log(n_array)), np.max(np.log(n_array)), 20)
plt.plot(np.log(n_array), np.log(autocorrtime), '.')
plt.plot(xrange, fitf(xrange, *popt))
plt.title(r'$log(\tau)$ vs. log(Lattice size)')
plt.xlabel('log(Lattice size)')
plt.ylabel(r'$log(\tau)')
plt.show()

# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

# worm algorithm
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

def calculate_energy(lattice, i, j):
    """Calculate the energy cost of flipping the spin at site (i,j)"""
    size = lattice.shape[0]
    spin = lattice[i, j]
    neighbours = lattice[(i+1)%size, j] + lattice[i, (j+1)%size] + lattice[(i-1)%size, j] + lattice[i, (j-1)%size]
    return 2 * spin * neighbours

def worm_algorithm(lattice, iterations, burnin, beta):
    """Simulate the 2D Ising model using the worm algorithm"""
    size = lattice.shape[0]
    tot_spins = np.zeros(iterations)
    tot_energy = np.zeros(iterations)
    
    worm = set()  # Set of lattice coordinates in the worm
    num_accepted = 0  # Counter for the number of accepted moves
    # Choose a random site to start the worm
    i, j = random.randint(0, size-1), random.randint(0, size-1)
    worm.add((i, j))  # Add the site to the worm
    # Perform iterations steps of the algorithm
    for step in range(iterations+burnin):
        # Choose a random direction to move the head or tail of the worm
        if random.random() < 0.5:
            # Move the head of the worm
            if worm:
                i, j = random.choice(list(worm))
            else:
                # If the worm is empty, choose a random point on the lattice
                i, j = random.randint(0, size-1), random.randint(0, size-1)
            neighbours = [(i+1)%size, j], [(i-1)%size, j], [i, (j+1)%size], [i, (j-1)%size]
            # Calculate the energy cost of moving to each neighboring site
            energies = [calculate_energy(lattice, x, y) for x, y in neighbours]
            # Choose a site to move to with probability proportional to exp(-beta*energy)
            probs = [np.exp(-beta*energy) for energy in energies]
            prob_sum = sum(probs)
            probs = [prob/prob_sum for prob in probs]  # Normalize the probabilities
            k = np.random.choice(4, p=probs)
            new_i, new_j = neighbours[k]
            # Calculate the energy cost of adding or removing the new site
            add_energy = calculate_energy(lattice, new_i, new_j)
            remove_energy = calculate_energy(lattice, i, j)
            # Calculate the acceptance probability for the move
            p_accept = np.exp(-beta * (add_energy - remove_energy))
            if random.random() < p_accept:
                # Move the head of the worm to the new site and add it to the worm
                worm.add((new_i, new_j))
                num_accepted += 1
        else:
            if worm:
                # Move the tail of the worm
                i, j = random.choice(list(worm))
                worm.remove((i, j))
                # Calculate the energy cost of removing the site
                energy = calculate_energy(lattice, i, j)
                p_remove = 1 - np.exp(-beta * energy)
                if random.random() < p_remove:
                    # Remove the site from the worm
                    num_accepted += 1
            else:
                # If the worm is empty, choose a random point on the lattice
                i, j = random.randint(0, size-1), random.randint(0, size-1)
                worm.add((i, j))

        # Flip the spin at the head of the worm
        lattice[i, j] *= -1

        if step >= burnin:
            tot_spins[step - burnin] = np.sum(lattice)
            energyapp = energy_toroid(lattice, 1, 0)
            tot_energy[step - burnin] = energyapp
    
    return tot_spins, tot_energy

spinsworm, energyworm = worm_algorithm(lattice_p, 100000, 10000, 1)

plt.plot(spinsworm/(N*N))
plt.show()
plt.plot(energyworm)
plt.show()