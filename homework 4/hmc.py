import numpy as np
import matplotlib.pyplot as plt
from leapfrog import leapfrog, ham, fitfun


def HMC(phi0, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, hmcn_samples, hmcn_burn, tau = 1, beta = 1000):
    acceptance = 0
    phi_all0 = np.zeros((hmcn_samples))
    phi_all1 = np.zeros((hmcn_samples))
    phi_all2 = np.zeros((hmcn_samples))
    phi = phi0.copy()
    for _ in range(hmcn_burn):
        #p0 = np.random.normal(1, 0)
        p0 = np.array([np.random.normal(1,0), np.random.normal(1,0), np.random.normal(1,0)])
        p_leap, phi_leap = leapfrog(p0, phi, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, tau, beta)
        H_old = ham(p0, phi, hmcmpik, hmcfk, hmcdeltafk, beta)
        H_new = ham(p_leap, phi_leap, hmcfk, hmcdeltafk, beta)
        delta_E = H_new - H_old
        if np.random.rand() < np.exp(-delta_E):
            phi = phi_leap
        else:
            phi = phi

    for i in range(hmcn_samples):
        #p0 = np.random.normal(1, 0)
        p0 = np.array([np.random.normal(1,0), np.random.normal(1,0), np.random.normal(1,0)])
        p_leap, phi_leap = leapfrog(p0, phi, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, tau, beta)
        H_old = ham(p0, phi, hmcmpik, hmcfk, hmcdeltafk, beta)
        H_new = ham(p_leap, phi_leap, hmcfk, hmcdeltafk, beta)
        delta_E = H_new - H_old
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

nmd = 100
Tau = 1
N_samples = 1000000
N_burn = 100000


phi_hmc = HMC(phidata, mpik, fk, deltafk, nmd, N_samples, N_burn, Tau)

# create x values to plot the behavior of φi as a function of HMC trajectory
hmc_trajectory = np.arange(0, N_samples, 1) 

#calculate the average values of phi_i and their std in Markov chain,
# which is the best fit values and its fit error
average_phi0 = np.mean(phi_hmc[0])
average_phi1 = np.mean(phi_hmc[1])
average_phi2 = np.mean(phi_hmc[2])

std_phi0 = np.std(phi_hmc[0])
std_phi1 = np.std(phi_hmc[1])
std_phi2 = np.std(phi_hmc[2])




fig = plt.figure(figsize=(10,7.5))

plt.plot(hmc_trajectory, phi_hmc[0],'.', label='phi_0')
plt.plot(hmc_trajectory, phi_hmc[1],'.', label='phi_1')
plt.plot(hmc_trajectory, phi_hmc[2],'.', label='phi_2')
plt.xlabel('HMC tajectory')
plt.ylabel('phi_i')
plt.grid()
plt.legend()
plt.show()

