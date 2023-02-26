#!/usr/bin/python
# libraries
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.optimize import curve_fit
from scipy import special
from mpmath import mp
from numba import njit
import time
from tqdm import tqdm
from error_analysis import bootstrap

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
# function to calculate energy in flat geometry is now depreciated and can be found in ./obsolete.py/

# the first line where the type is asserted explicitly during numba call,
# doesn't work when the metropolis function returns the final spin config
# numba doesn't support type asserting 2D arrays (?)
# but using the second line instead of the first, speeds up the code during the first function call
# there is no difference in the consecutive function calls

#@njit("UniTuple(f8[:], 2)(f8[:,:], i8, i8, f8, f8, f8, f8)", nogil=True)
@njit(nogil=True)
def metropolis(spin_config, iterations, burnin, J, h, beta, energy):
    '''
    Metropolis algorithm for a 2D Ising Model.
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
    susceptibility = np.zeros(iterations)

    for i in range(iterations + burnin):
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
            susceptibility[i - burnin] = beta*(np.var(spin_config))*(N*N)
    # return the arrays of total spins and energies over the iterations and the final spin config 
    return tot_spins, tot_energy, tot_spins_noburnin, tot_energy_noburnin, susceptibility

#function to calculate the spin autocorrelation time, given an array of net spins about MC time
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

def susceptibility(data,beta,n_spin,Nbst=1000):
    Nconf = data.shape
    sampleMeans = np.zeros(Nbst)
    
    for k in range(Nbst):
        idx = np.random.randint(0,Nconf,size=Nconf)
        
        sample = data[idx]
        
        sampleMeans[k] = (beta * ( np.mean(sample**2) - np.mean(sample)**2 )) / (n_spin**2)
        
    return np.mean(sampleMeans), np.std(sampleMeans)


def spesificHeat(data,beta,n_spin,Nbst=1000):
     
    Nconf = data.shape
    sampleMeans = np.zeros(Nbst)
    
    for k in range(Nbst):
        idx = np.random.randint(0,Nconf,size=Nconf)
        
        sample = data[idx]
        
        sampleMeans[k] = (beta**2 * ( np.mean(sample**2) - np.mean(sample)**2 )) / (n_spin**2)
        
    return np.mean(sampleMeans), np.std(sampleMeans)    

# test if everything's working as expected
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-
def test():
    N = 20
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
    spins_n, energies_n, spinsnob_n, energiesnob_n, sus_n = metropolis(lattice_n, iter, burn, j, h, beta, energy_n)
    spins_p, energies_p, spinsnob_p, energiesnob_p, sus_p = metropolis(lattice_p, iter, burn, j, h, beta, energy_p)
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

# algorithm behaviour
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-
def algobehave():
    n1 = 50
    n2 = 70
    n3 = 90

    init_random = np.random.random((n1, n1))
    init_spin1 = np.zeros((n1, n1))
    init_spin1[init_random >= .8] = -1
    init_spin1[init_random < .8] = 1

    init_random = np.random.random((n2, n2))
    init_spin2 = np.zeros((n2, n2))
    init_spin2[init_random >= .8] = -1
    init_spin2[init_random < .8] = 1

    init_random = np.random.random((n3, n3))
    init_spin3 = np.zeros((n3, n3))
    init_spin3[init_random >= .8] = -1
    init_spin3[init_random < .8] = 1

    energy1 = energy_toroid(init_spin1, j, h)
    energy2 = energy_toroid(init_spin2, j, h)
    energy3 = energy_toroid(init_spin3, j, h)

    spins1, energies1, spinsnob1, energiesnob1, sus1 = metropolis(init_spin1, iter, burn, j, h, beta, energy1)
    spins2, energies2, spinsnob2, energiesnob2, sus2 = metropolis(init_spin2, iter, burn, j, h, beta, energy2)
    spins3, energies3, spinsnob3, energiesnob3, sus3 = metropolis(init_spin3, iter, burn, j, h, beta, energy3)


    plt.plot(spins1/(n1*n1), label=f'N = {n1}')
    plt.plot(spins2/(n2*n2), label=f'N = {n2}')
    plt.plot(spins3/(n3*n3), label=f'N = {n3}')
    plt.xlabel('Algorithm Time Steps')
    plt.ylabel('Net Magnetisation (<M>) [J=1]')
    plt.title('Net Magnetisation vs. Algorithm Time Steps')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

    plt.plot(energies1/(n1*n1), label=f'N = {n1}')
    plt.plot(energies2/(n2*n2), label=f'N = {n2}')
    plt.plot(energies3/(n3*n3), label=f'N = {n3}')
    plt.xlabel('Algorithm Time Steps')
    plt.ylabel('Energy per site [J=1]')
    plt.title('Energy per site vs. Algorithm Time Steps')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()

# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

# mag phase transition
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-
def magphasetrans():
    beta_arr = np.linspace(0, 1, 70)

    n1 = 30
    n2 = 50
    n3 = 70

    init_random = np.random.random((n1, n1))
    init_spin1 = np.zeros((n1, n1))
    init_spin1[init_random >= .8] = -1
    init_spin1[init_random < .8] = 1

    init_random = np.random.random((n2, n2))
    init_spin2 = np.zeros((n2, n2))
    init_spin2[init_random >= .8] = -1
    init_spin2[init_random < .8] = 1

    init_random = np.random.random((n3, n3))
    init_spin3 = np.zeros((n3, n3))
    init_spin3[init_random >= .8] = -1
    init_spin3[init_random < .8] = 1

    energy1 = energy_toroid(init_spin1, j, h)
    energy2 = energy_toroid(init_spin2, j, h)
    energy3 = energy_toroid(init_spin3, j, h)

    netmag1 = np.zeros(len(beta_arr))
    netmag2 = np.zeros(len(beta_arr))
    netmag3 = np.zeros(len(beta_arr))

    netmag1_err = np.zeros(len(beta_arr))
    netmag2_err = np.zeros(len(beta_arr))
    netmag3_err = np.zeros(len(beta_arr))

    '''
    for i in tqdm(range(len(netmag1))):
        totspin1, totenergy1, totspinnob1, totenergynob1, sus1 = metropolis(init_spin1, iter, burn, j, h, beta_arr[i], energy1)
        netmag1[i] = np.average(totspin1)/(n1*n1)

    for i in tqdm(range(len(netmag2))):
        totspin2, totenergy2, totspinnob2, totenergynob2, sus2 = metropolis(init_spin2, iter, burn, j, h, beta_arr[i], energy2)
        netmag2[i] = np.average(totspin2)/(n2*n2)

    for i in tqdm(range(len(netmag3))):
        totspin3, totenergy3, totspinnob3, totenergynob3, sus3 = metropolis(init_spin3, iter, burn, j, h, beta_arr[i], energy3)
        netmag3[i] = np.average(totspin3)/(n3*n3)
    '''
       
    for i in tqdm(range(len(netmag1))):  
        totspin1, totenergy1, totspinnob1, totenergynob1, sus1 = metropolis(init_spin1, iter, burn, j, h, beta_arr[i], energy1)
        #netmag1[i] = np.average(totspin1)/(n1*n1)
        # error_calculation bootstrap
        totspin1_mean, totspin1_err = bootstrap(totspin1,num_bs)
        netmag1[i] = totspin1_mean/(n1*n1)
        netmag1_err[i] = totspin1_err/(n1*n1)
         
    for i in tqdm(range(len(netmag2))):
        totspin2, totenergy2, totspinnob2, totenergynob2, sus2 = metropolis(init_spin2, iter, burn, j, h, beta_arr[i], energy2)
        #netmag2[i] = np.average(totspin2)/(n2*n2)
        totspin2_mean, totspin2_err = bootstrap(totspin2,num_bs)
        netmag2[i] = totspin2_mean/(n2*n2)
        netmag2_err[i] = totspin2_err/(n2*n2)

    for i in tqdm(range(len(netmag3))):
        totspin3, totenergy3, totspinnob3, totenergynob3, sus3 = metropolis(init_spin3, iter, burn, j, h, beta_arr[i], energy3)
        #netmag3[i] = np.average(totspin3)/(n3*n3)
        totspin3_mean, totspin3_err = bootstrap(totspin3,num_bs)
        netmag3[i] = totspin3_mean/(n3*n3)
        netmag3_err[i] = totspin3_err/(n3*n3)
    
    # critcal coupling J_c
    J_c = (1/2) * np.log(1 + np.sqrt(2))

    # abs magnetization per site with h=0 in thermodynamic limit
    def abs_mag(J):
        if J > J_c:
            return (1 - (1/np.sinh(2*J)**4))**(1/8)
        else:
            return 0

    def K(m):
        return special.ellipk(m)

    # energy per site with h=0
    def e(J,m):
        return - J * mp.coth(2*J) * ( 1 + (2/np.pi)*(2*np.tanh(2*J)**2 - 1)*K(m)*(4*mp.sech(2*J)**2 * np.tanh(2*J)**2)) 

    betaan = np.linspace(0.01, 1, 50)
    energyan = np.zeros(len(betaan))
    magan = np.zeros(len(betaan))
    # loop over J and calculate energy and abs_magnetization
    for i in range(len(betaan)):
        m = abs_mag(betaan[i]) 
        magan[i] = m
        energyan[i] = e(betaan[i],m)

    #plot the results
    plt.errorbar(beta_arr, netmag1, netmag1_err, fmt='.', capthick=1, label=f'N = {n1}')
    plt.errorbar(beta_arr, netmag2, netmag2_err, fmt='.', capthick=1, label=f'N = {n2}')
    plt.errorbar(beta_arr, netmag3, netmag3_err, fmt='.', capthick=1, label=f'N = {n3}')
    #plt.plot(beta_arr, netmag1, '.', label=f'N = {n1}')
    #plt.plot(beta_arr, netmag2, '.', label=f'N = {n2}')
    #plt.plot(beta_arr, netmag3, '.', label=f'N = {n3}')
    plt.plot(betaan, magan, label='Analytical')
    plt.xlabel(r'Inverse Temperature ($\beta$)')
    plt.ylabel(r'Net Magnetisation (<M>) [J = 1]')
    plt.title('Net Magnetisation vs. Inverse Temperature')
    plt.legend(loc='upper left')
    plt.show()
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

# chi phase transition
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-
def chiphasetrans():
    beta_arr = np.linspace(0, 1, 50)

    n1 = 10
    n2 = 30
    n3 = 60

    init_random = np.random.random((n1, n1))
    init_spin1 = np.zeros((n1, n1))
    init_spin1[init_random >= .8] = -1
    init_spin1[init_random < .8] = 1

    init_random = np.random.random((n2, n2))
    init_spin2 = np.zeros((n2, n2))
    init_spin2[init_random >= .8] = -1
    init_spin2[init_random < .8] = 1

    init_random = np.random.random((n3, n3))
    init_spin3 = np.zeros((n3, n3))
    init_spin3[init_random >= .8] = -1
    init_spin3[init_random < .8] = 1

    energy1 = energy_toroid(init_spin1, j, h)
    energy2 = energy_toroid(init_spin2, j, h)
    energy3 = energy_toroid(init_spin3, j, h)

    chi1 = np.zeros(len(beta_arr))
    chi2 = np.zeros(len(beta_arr))
    chi3 = np.zeros(len(beta_arr))

    for i in tqdm(range(len(chi1))):
        totspin1, totenergy1, totspinnob1, totenergynob1, sus1 = metropolis(init_spin1, iter, burn, j, h, beta_arr[i], energy1)
        chi1[i] = np.average(sus1)

    for i in tqdm(range(len(chi2))):
        totspin2, totenergy2, totspinnob2, totenergynob2, sus2 = metropolis(init_spin2, iter, burn, j, h, beta_arr[i], energy2)
        chi2[i] = np.average(sus2)

    for i in tqdm(range(len(chi3))):
        totspin3, totenergy3, totspinnob3, totenergynob3, sus3 = metropolis(init_spin3, iter, burn, j, h, beta_arr[i], energy3)
        chi3[i] = np.average(sus3)

    #plot the results
    plt.plot(beta_arr, chi1, '.', label=f'N = {n1}')
    plt.plot(beta_arr, chi2, '.', label=f'N = {n2}')
    plt.plot(beta_arr, chi3, '.', label=f'N = {n3}')
    plt.xlabel(r'Inverse Temperature ($\beta$)')
    plt.ylabel(r'Susceptibility ($\chi$)')
    plt.title('Susceptibility vs. Inverse Temperature')
    plt.legend(loc='upper left')
    plt.show()
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-


###### susceptibility and spesific heat

def criticalEx():
    beta_arr = np.linspace(0, 1, 70)

    n1 = 10
    n2 = 30
    n3 = 50

    init_random = np.random.random((n1, n1))
    init_spin1 = np.zeros((n1, n1))
    init_spin1[init_random >= .8] = 1
    init_spin1[init_random < .8] = -1

    init_random = np.random.random((n2, n2))
    init_spin2 = np.zeros((n2, n2))
    init_spin2[init_random >= .8] = 1
    init_spin2[init_random < .8] = -1

    init_random = np.random.random((n3, n3))
    init_spin3 = np.zeros((n3, n3))
    init_spin3[init_random >= .8] = 1
    init_spin3[init_random < .8] = -1

    energy1 = energy_toroid(init_spin1, j, h)
    energy2 = energy_toroid(init_spin2, j, h)
    energy3 = energy_toroid(init_spin3, j, h)

    suscep1 = np.zeros(len(beta_arr))
    suscep2 = np.zeros(len(beta_arr))
    suscep3 = np.zeros(len(beta_arr))

    suscep1_err = np.zeros(len(beta_arr))
    suscep2_err = np.zeros(len(beta_arr))
    suscep3_err = np.zeros(len(beta_arr))

    speH1 = np.zeros(len(beta_arr))
    speH2 = np.zeros(len(beta_arr))
    speH3 = np.zeros(len(beta_arr))

    speH1_err = np.zeros(len(beta_arr))
    speH2_err = np.zeros(len(beta_arr))
    speH3_err = np.zeros(len(beta_arr))

    
    for i in tqdm(range(len(speH1))):
        totspin1, totenergy1, totspinnob1, totenergynob1, sus1 = metropolis(init_spin1, iter, burn, j, h, beta_arr[i], energy1)
        suscep1[i], suscep1_err[i] = susceptibility(totspin1,beta_arr[i],n1) 
        speH1[i],speH1_err[i] = spesificHeat(totenergy1,beta_arr[i],n1) 

    
    for i in tqdm(range(len(speH2))):
        totspin2, totenergy2, totspinnob2, totenergynob2, sus2 = metropolis(init_spin2, iter, burn, j, h, beta_arr[i], energy2)
        suscep2[i], suscep2_err[i] = susceptibility(totspin2,beta_arr[i],n2) 
        speH2[i],speH2_err[i] = spesificHeat(totenergy2,beta_arr[i],n2)  
    

    for i in tqdm(range(len(speH3))):
        totspin3, totenergy3, totspinnob3, totenergynob3, sus3 = metropolis(init_spin3, iter, burn, j, h, beta_arr[i], energy3)
        suscep3[i], suscep3_err[i] = susceptibility(totspin3,beta_arr[i],n3) 
        speH3[i],speH3_err[i] = spesificHeat(totenergy3,beta_arr[i],n3) 

###### analytical spesific heat
    def specificHeat_exact(J):
        k = ( 2*np.sinh(2*J) / np.cosh(2*J)**2 )
        return (4*J**2/(np.pi*np.tanh(2*J)**2)) * ( special.ellipkinc(np.pi/2,k**2) - special.ellipeinc(np.pi/2,k**2) - (1-np.tanh(2*J)**2)*((np.pi/2)+(2*np.tanh(2*J)**2-1)*special.ellipkinc(np.pi/2,k**2)) )   
        

    betaan = np.linspace(0.01, 1, 50)
    Heatan = np.zeros(len(betaan))
    # loop over J and calculate specific heat
    for i in range(len(betaan)):
        Heatan[i] = specificHeat_exact(betaan[i]) 

    #plot the results
    plt.errorbar(beta_arr, speH1, speH1_err, fmt='.', capthick=1, label=f'N = {n1}')
    plt.errorbar(beta_arr, speH2, speH2_err, fmt='.', capthick=1, label=f'N = {n2}')
    plt.errorbar(beta_arr, speH3, speH3_err, fmt='.', capthick=1, label=f'N = {n3}')
    plt.plot(betaan, Heatan, label='Analytical')
    plt.xlabel(r'Inverse Temperature ($\beta$)')
    plt.ylabel(r'Specific Heat (C)')
    plt.title('Specific Heat vs. Inverse Temperature')
    plt.legend(loc='upper left')
    plt.show()
    
    plt.errorbar(beta_arr, suscep1, suscep1_err, fmt='.', capthick=1, label=f'N = {n1}')
    plt.errorbar(beta_arr, suscep2, suscep2_err, fmt='.', capthick=1, label=f'N = {n2}')
    plt.errorbar(beta_arr, suscep3, suscep3_err, fmt='.', capthick=1, label=f'N = {n3}')
    plt.xlabel(r'Inverse Temperature ($\beta$)')
    plt.ylabel(r'Susceptibility')
    plt.title('Magnetic Susceptibility vs. Inverse Temperature')
    plt.legend(loc='upper left')
    plt.show()
#############################################


# energy autocorrelation
# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-
def dyncritexp():
    n_array = np.arange(5, 151, 2)

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

# -0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-

# init values
j = 1.
h = 0.
beta = 1.
iter = 100000
burn = 30000
num_bs = 2000 # number of bootstrap samples

if __name__ == '__main__':
    # uncomment the next lines to run a specific part of the code
    #test()
    #algobehave()
    #magphasetrans()
    #chiphasetrans()
    #criticalEx()
    #dyncritexp()
    pass