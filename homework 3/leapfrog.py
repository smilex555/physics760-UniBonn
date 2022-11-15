import numpy as np
import matplotlib.pyplot as plt

def leapfrog(p, phi, nmd, N = 20, beta = 1, jn = 1, h = 1): #p and phi are some initial values
    #nmd is the number of steps
    #N, beta, jn, h are parameters of the system
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

def hamlf(p, phi, N = 20, beta = 1, jn = 1, h = 1): #function to calculate the energy of a given configuration
    return 0.5*p*p + 0.5*phi*phi/(beta*jn) - N*np.log(2*np.cosh(beta*h + phi))

p, phi = 1, 1 #fixed init values
#p, phi = np.random.rand(), np.random.rand() #random init values
H_old = hamlf(p, phi) #energy of the init configuration
H_new = np.zeros(100) #array to store the energies of the final config
for i in range(100):
    p2, phi2 = leapfrog(p, phi, i+1)
    H_new[i] = hamlf(p2, phi2)

xrange = np.arange(100) + 1
yval = np.abs((H_new - H_old)/H_old)

#plotting
plt.plot(xrange, yval, '.')
plt.yscale('log')
plt.xlabel('MD steps')
plt.ylabel('$\\vert\\frac{\\mathcal{H}[p_f, \\phi_f] - \\mathcal{H}[p_i, \\phi_i]}{\\mathcal{H}[p_i, \\phi_i]}\\vert$')
plt.title('Relative error vs. MD steps')
plt.grid()
plt.show()