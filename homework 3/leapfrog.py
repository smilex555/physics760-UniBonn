import numpy as np

def leapfrog(p, phi, nmd, beta = 1, jn = 1, h = 1): #p and phi are some initial values
    #nmd is the number of steps
    #beta, jn, h are parameters of the system
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