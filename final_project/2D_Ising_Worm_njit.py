#!/usr/bin/python
# libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
from tqdm import tqdm

# function to calculate energy in a toroidal geometry, i.e.,
# lattice points on the boundary have lattice points from the opposite edges as neighbours
def energy_toroid(spin_config, J, h):
    '''
    Calculate total energy of a given spin config in toroidal geomtery, i.e.,
    lattice points on the edges have lattice points on the opposite edge as neighbours.
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

@njit(nogil=True)
def justburn(spin_config, burnin, J, h, beta, energy):
    N = len(spin_config)
    spin_config = spin_config.copy()
    
    for _ in range(burnin):
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

    # return the arrays of total spins and energies over the iterations and the final spin config 
    return spin_config, energy

@njit(nogil=True)
def worm(spin_config, iterations, burnin, J, h, beta, energy):
    N = len(spin_config)
    tot_spins = np.zeros(iterations)
    tot_energy = np.zeros(iterations)
    #insert burnin
    spin_config, energy = justburn(spin_config, burnin, J, h, beta, energy)

    spin_config = spin_config.copy()
    for step in range(iterations):
        x, y = np.random.randint(N), np.random.randint(N)
        worm = np.array([[x, y]])
        tail = np.array([[x, y]])
        neighbours = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        while True:
            # move the head
            while True:
                dir = np.random.choice(np.array([-1, 0, 1]), 2)
                if np.any((neighbours[:, 0] == dir[0])*(neighbours[:, 1] == dir[1])): break
            xnew, ynew = x+dir[0], y+dir[1]
            codi = np.array([[xnew, ynew]])
            if (xnew > N-1) or (ynew > N-1) or (xnew < 0) or (ynew < 0): break
            elif (xnew == tail[0, 0])*(ynew == tail[0, 1]): break
            elif spin_config[xnew, ynew] == spin_config[x, y]:
                if not np.any((worm[:, 0] == codi[0, 0])*(worm[:, 1] == codi[0, 1])):
                    worm = np.append(worm, codi, axis = 0)
                    x, y = xnew, ynew
                else:
                    index = np.where((worm[:, 0] == codi[0, 0])*(worm[:, 1] == codi[0, 1]))[0]
                    if index[0] == len(worm) - 2:
                        x, y = worm[index[0]][0], worm[index[0]][1]
                    else:
                        if np.random.random() < 0.5:
                            x, y = worm[index[0] - 1][0], worm[index[0] - 1][1]
                        else:
                            x, y = worm[index[0] + 1][0], worm[index[0] + 1][1]
            else:
                # check energy cost of flipping
                spin_i = spin_config[xnew, ynew]
                spin_f = -1*spin_i
                E_i = 0
                E_f = 0
                E_i += -J*spin_i*spin_config[(xnew-1)%N, ynew]
                E_i += -J*spin_i*spin_config[(xnew+1)%N, ynew]
                E_i += -J*spin_i*spin_config[xnew, (ynew-1)%N]
                E_i += -J*spin_i*spin_config[xnew, (ynew+1)%N]
                E_i += -h*spin_i
                E_f += -J*spin_f*spin_config[(xnew-1)%N, ynew]
                E_f += -J*spin_f*spin_config[(xnew+1)%N, ynew]
                E_f += -J*spin_f*spin_config[xnew, (ynew-1)%N]
                E_f += -J*spin_f*spin_config[xnew, (ynew+1)%N]
                E_f += -h*spin_f
                dE = E_f - E_i
                if (dE>0)*(np.random.random() < np.exp(-beta*dE)):
                    energy += dE
                    worm = np.append(worm, codi, axis = 0)
                    spin_config[xnew, ynew] = spin_f
                    x, y = xnew, ynew
                elif dE<=0:
                    spin_config[xnew, ynew] = spin_f
                    energy += dE
                    worm = np.append(worm, codi, axis = 0)
                    x, y = xnew, ynew
                else: break
        
        tot_spins[step] = np.sum(spin_config)
        tot_energy[step] = energy 

    return tot_spins, tot_energy

def spin_autocorr_time(spins):
    '''
    Calculate the spin autocorrelation time for an Ising model from a series of net spin values.
    Args:
        spins (1D numpy.ndarray): 1D NumPy array of net spin values
    Returns:
        tau (float): Spin autocorrelation time
    '''
    # Compute the mean and variance of the spin series
    m = np.mean(spins)
    v = np.var(spins)
    if v == 0.:
        return np.NaN

    # Compute the autocorrelation function of the spin series
    acf = np.correlate(spins - m, spins - m, mode='full')
    acf = acf[len(acf)//2:] / (v * (len(spins) - 1))

    # Find the first index where the autocorrelation function is less than exp(-1)
    idx = np.where(acf < np.exp(-1))[0][0]

    # Compute the spin autocorrelation time
    tau = 2 * np.sum(acf[:idx])

    return tau

latsize = 20

init_spin = np.random.choice([1, -1], (latsize, latsize))

energy_t = energy_toroid(init_spin, 1., 0.)
spins, nrg = worm(init_spin, 70000, 30000, 1., 0, 1., energy_t)

plt.plot(spins/(latsize*latsize))
plt.show()

def dyncritexp():
    n_array = np.arange(5, 151, 2)

    autocorrtime = np.zeros(len(n_array))
    for i in tqdm(range(len(n_array))):
        init_random = np.random.random((n_array[i], n_array[i]))
        lattice = np.zeros((n_array[i], n_array[i]))
        lattice[init_random <= .8] = 1
        lattice[init_random > .8] = -1
        energy_t = energy_toroid(lattice, 1, 0)
        totspin, totenergy = worm(lattice, 100000, 0, 1, 0, 1, energy_t)
        autocorrtime[i] = spin_autocorr_time(totspin)


    # fit to verify tau-lattice size scaling
    def fitf(x, m, c):
        return m*x + c

    if np.all(np.isnan(autocorrtime)):
        print('Not enough Iterations!')

    if not np.all(np.isnan(autocorrtime)):
        popt, pcov = curve_fit(fitf, np.log(n_array), np.log(autocorrtime))
        print('Fit slope:', np.round(popt[0], 2))

        # plot the results
        xrange = np.linspace(np.min(np.log(n_array)), np.max(np.log(n_array)), 20)
        plt.plot(np.log(n_array), np.log(autocorrtime), '.', label='data')
        plt.plot(xrange, fitf(xrange, *popt), label='fit')
        plt.title(r'$log(\tau)$ vs. log(Lattice size)')
        plt.xlabel('log(Lattice size)')
        plt.ylabel(r'$log(\tau)$')
        plt.legend()
        plt.grid()
        plt.show()

if __name__=='__main__':
    dyncritexp()
    pass