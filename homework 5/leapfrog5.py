import numpy as np
import matplotlib.pyplot as plt

def leapfrog(p, phi, nmd, tau = 1):
    eps = tau/nmd
    fpi, fphi = p, phi
    # first step
    fphi += .5*eps*fpi
    # intermediate steps
    for _ in range(nmd - 1):
        fpi -= eps*2*fphi
        fphi += eps*fpi
    # last step
    fpi -= eps*2*fphi
    fphi += .5*eps*fpi
    return fpi, fphi

def ham(p, phi):
    return 0.5*np.power(p,2) + np.power(phi,2)

# p, phi = np.array([1., 1., 1.]), np.array([800., 800., 600.]) #fixed init values
p, phi = np.random.rand(), np.random.rand() #random init values
H_old = ham(p, phi) # energy of the init configuration
nmdtotal = 100 # NMD steps max value
H_new = np.zeros(nmdtotal) # array to store the energies of the final config
for i in range(nmdtotal):
    p2, phi2 = leapfrog(p, phi, i+1)
    H_new[i] = ham(p2, phi2)

xrange = np.arange(nmdtotal)
yval = np.abs((H_new - H_old)/H_old)

# plotting
plt.plot(xrange, yval, '.')
plt.yscale('log')
plt.xlabel('MD steps')
plt.ylabel('$\\vert\\frac{\\mathcal{H}[p_f, \\phi_f] - \\mathcal{H}[p_i, \\phi_i]}{\\mathcal{H}[p_i, \\phi_i]}\\vert$')
plt.title('Relative error vs. MD steps')
plt.grid()
plt.show()