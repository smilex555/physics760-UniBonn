import numpy as np
import matplotlib.pyplot as plt
from leapfrog import leapfrog, ham

def HMC(phi0, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, hmcn_samples, hmcn_burn, tau = 1, beta = 1000):
    acceptance = 0
    phi_all = np.zeros((hmcn_samples))
    phi = phi0.copy()
    for _ in range(hmcn_burn):
        p0 = np.random.normal(1, 0)
        p_leap, phi_leap = leapfrog(p0, phi, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, tau, beta)
        H_old = ham(p0, phi, hmcmpik, hmcfk, hmcdeltafk, beta)
        H_new = ham(p_leap, phi_leap, hmcfk, hmcdeltafk, beta)
        delta_E = H_new - H_old
        if np.random.rand() < np.exp(-delta_E):
            phi = phi_leap
        else:
            phi = phi

    for i in range(hmcn_samples):
        p0 = np.random.normal(1, 0)
        p_leap, phi_leap = leapfrog(p0, phi, hmcmpik, hmcfk, hmcdeltafk, hmcnmd, tau, beta)
        H_old = ham(p0, phi, hmcmpik, hmcfk, hmcdeltafk, beta)
        H_new = ham(p_leap, phi_leap, hmcfk, hmcdeltafk, beta)
        delta_E = H_new - H_old
        if np.random.rand() < np.exp(-delta_E):
            phi = phi_leap
            acceptance += 1
        else:
            phi = phi

    return phi_all, acceptance/hmcn_samples