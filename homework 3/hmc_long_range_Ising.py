import numpy as np
import random
import scipy.stats as st
import matplotlib.pyplot as plt


def hamlf(p, phi, N = 20, jn = 1, h = 1): #function to calculate the energy of a given configuration
    beta = 1
    return 0.5*p*p + 0.5*phi*phi/(beta*jn) - N*np.log(2*np.cosh(beta*h + phi))

def leapfrog(p, phi, nmd, N = 20, jn = 1, h = 1): #p and phi are some initial values
    #nmd is the number of steps
    #N, beta, jn, h are parameters of the system
    beta = 1
    eps = 1/nmd
    pi, phhi = p, phi
    #first step
    phhi = phhi + 0.5*eps*pi
    #intermediate steps
    for i in range(nmd - 1):
        pi = pi - eps*(phhi/(beta*jn) - N*np.tanh(beta*h + phhi))
        phhi = phhi + eps*pi
    #last step
    pi = pi - eps*(phhi/(beta*jn) - N*np.tanh(beta*h + phhi))
    phhi = phhi + 0.5*eps*pi
    return pi, phhi #p-f, phi-f values


def HMC(phi0,J,h,N,nmd,N_samples,N_burn):
    phi = 1
    acceptence = 0
    MC=[]
    mu = 0
    sig = 1
    
    for i in range(N_burn):
        p0 = np.random.normal(mu, sig)
        p_leap, phi_leap = leapfrog(p0, phi0, nmd, N, J, h)
        H_old = hamlf(p0, phi0, N, J, h)
        H_new = hamlf(p_leap,phi_leap, N, J, h)
        delta_E = H_old - H_new
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random < np.exp(-delta_E):
            phi = phi_leap
        else:
            phi = phi0
    
    for i in range(N_samples):
        p0 = np.random.normal(mu, sig)
        p_leap, phi_leap = leapfrog(p0, phi0, nmd, N, J, h)
    
        H_old = hamlf(p0, phi0, N, J, h)
        H_new = hamlf(p_leap,phi_leap, N, J, h)
        delta_E = H_old - H_new
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random < np.exp(-delta_E):
            phi = phi_leap
            acceptence += 1
        else:
            phi = phi0
        
        MC[i] = phi
    
    return MC, acceptence/N_samples

'''
h = 0.5
N = 
J = 
nmd = 
N_samples =
N_burn =
'''





