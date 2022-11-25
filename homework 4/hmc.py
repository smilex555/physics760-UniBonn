import numpy as np
import matplotlib.pyplot as plt
#from leapfrog import leapfrog, ham, fitfun


def leapfrog(p, phi, mpik, fk, deltafk, nmd, tau, beta = 1000):
    eps = tau/nmd
    fpi, fphi = p.copy(), phi.copy()
    # first step
    fphi += .5*eps*fpi
    # intermediate steps
    for _ in range(nmd - 1):
        for i in range(len(phi)):
            fpi[i] += eps*beta*(np.sum((fk - fitfun(mpik, fphi))*(mpik**i)/(deltafk*deltafk)))
        fphi += eps*fpi
    # last step
    for i in range(len(phi)):
        fpi[i] += eps*beta*(np.sum((fk - fitfun(mpik, fphi))*(mpik**i)/(deltafk*deltafk)))
    fphi += .5*eps*fpi
    return fpi, fphi

def ham(p, phi, mpik, fk, deltafk, beta = 1000):
    return np.sum(0.5*p*p) + beta*0.5*np.sum((fk - fitfun(mpik, phi))*(fk - fitfun(mpik, phi))/(deltafk*deltafk))

def fitfun(mpi, x):
    return x[0] + x[1]*mpi + x[2]*mpi*mpi


def HMC(phi0, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, hmcn_samples, hmcn_burn, tau, beta = 1000):
    acceptance = 0
    phi_all0 = np.zeros((hmcn_samples))
    phi_all1 = np.zeros((hmcn_samples))
    phi_all2 = np.zeros((hmcn_samples))
    phi = phi0.copy()
    for _ in range(hmcn_burn):
        #p0 = np.random.normal(1, 0)
        p0 = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        p_leap, phi_leap = leapfrog(p0, phi, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, tau, beta)
        H_old = ham(p0, phi, hmcmpik, hmcfk, hmcdeltafk, beta)
        H_new = ham(p_leap, phi_leap, hmcmpik, hmcfk, hmcdeltafk, beta)
        delta_E = H_new - H_old
        if np.random.rand() < np.exp(-delta_E):
            phi = phi_leap
        else:
            phi = phi

    for i in range(hmcn_samples):
        #p0 = np.random.normal(1, 0)
        p0 = np.array([np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,1)])
        #p0 = np.array([np.random.rand(), np.random.rand(), np.random.rand()])
        #p0 = np.array([1., 1., 1.])
        #print(p0)
        #print(phi)
        p_leap, phi_leap = leapfrog(p0, phi, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, tau, beta)
        #print(p_leap, phi_leap)
        H_old = ham(p0, phi, hmcmpik, hmcfk, hmcdeltafk, beta)
        #print(H_old)
        H_new = ham(p_leap, phi_leap, hmcmpik, hmcfk, hmcdeltafk, beta)
        #print(H_new)
        delta_E = np.abs((H_new - H_old))
        #print(delta_E)
        if np.random.rand() < np.exp(-delta_E):
            phi = phi_leap
            acceptance += 1
        else:
            phi = phi

        phi_all0[i] = phi[0]
        phi_all1[i] = phi[1]
        phi_all2[i] = phi[2]

    return phi_all0, phi_all1, phi_all2, acceptance/hmcn_samples

mpik = np.array([.176, .234, .260, .284, .324])
fk = np.array([960., 1025., 1055., 1085., 1130.])
deltafk = np.array([25., 20., 15., 10., 8.])

phidata = np.array([800., 800., 600.])

nmd = 20
Tau = 1
N_samples = 1000000
N_burn = 100000


phi_hmc = HMC(phidata, mpik, fk, deltafk, nmd, N_samples, N_burn, Tau)
print(phi_hmc[3])
# create x values to plot the behavior of φi as a function of HMC trajectory
hmc_trajectory = np.arange(0, N_samples, 1) 

#calculate the average values of phi_i and their std in Markov chain,
# which is the best fit values and its fit error
average_phi = np.mean(phi_hmc)
std_phi0 = np.std(phi_hmc)

# calculate fit function values with average phi values 
phi_fit_data = fitfun(mpik, average_phi)

plt.figure(figsize=(10,7.5))

plt.plot(hmc_trajectory, phi_hmc[0],'.', label='phi_0')
plt.plot(hmc_trajectory, phi_hmc[1],'.', label='phi_1')
plt.plot(hmc_trajectory, phi_hmc[2],'.', label='phi_2')
plt.xlabel('HMC tajectory')
plt.ylabel('phi_i')
plt.grid()
plt.legend()
plt.show()


plt.errorbar(mpik, fk, deltafk, fmt='.', capthick=1, label='neutron mass data')
plt.errorbar(mpik, phi_fit_data, 0, fmt='.', capthick=1, label='neutron mass fit')
plt.xlabel('pion mass [GeV]')
plt.ylabel('neutron mass [MeV]')
plt.grid()
plt.legend()
plt.show()