import numpy as np
import matplotlib.pyplot as plt

def leapfrog(p, phi, mpik, fk, deltafk, nmd, tau = 1, beta = 1000):
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

mpik = np.array([.176, .234, .260, .284, .324])
fk = np.array([960., 1025., 1055., 1085., 1130.])
deltafk = np.array([25., 20., 15., 10., 8.])

p, phi = np.array([1., 1., 1.]), np.array([800., 800., 600.]) #fixed init values
# p, phi = np.array([np.random.rand(), np.random.rand(), np.random.rand()]), np.array([np.random.rand(), np.random.rand(), np.random.rand()]) #random init values
H_old = ham(p, phi, mpik, fk, deltafk) # energy of the init configuration
nmdtotal = 100 # NMD steps max value
H_new = np.zeros(nmdtotal) # array to store the energies of the final config
for i in range(nmdtotal):
    p2, phi2 = leapfrog(p, phi, mpik, fk, deltafk, i+1)
    H_new[i] = ham(p2, phi2, mpik, fk, deltafk)

xrange = np.arange(100) + 1
yval = np.abs((H_new - H_old)/H_old)

# plotting
plt.plot(xrange, yval, '.')
plt.yscale('log')
plt.xlabel('MD steps')
plt.ylabel('$\\vert\\frac{\\mathcal{H}[p_f, \\phi_f] - \\mathcal{H}[p_i, \\phi_i]}{\\mathcal{H}[p_i, \\phi_i]}\\vert$')
plt.title('Relative error vs. MD steps')
plt.grid()
plt.show()