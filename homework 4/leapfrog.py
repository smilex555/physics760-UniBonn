import numpy as np
import matplotlib.pyplot as plt

def leapfrog(p, phi, mpik, fk, deltafk, nmd, tau = 1, beta = 1000):
    eps = tau/nmd
    #fpi, fphi = p, phi
    fpi, fphi = p.copy(), phi.copy()
    # first step
    fphi += .5*eps*fpi
    # intermediate steps
    for _ in range(nmd - 1):
        for i in range(len(phi)):
            fpi[i] += eps*beta*(np.sum((fk - fitfun(mpik, fphi))*(mpik**i)/(deltafk*deltafk)))
        fphi += eps*fpi
    #last step
    for i in range(len(phi)):
        fpi[i] += eps*beta*(np.sum((fk - fitfun(mpik, fphi))*(mpik**i)/(deltafk*deltafk)))
    fphi += .5*eps*fpi
    return fpi, fphi

def ham(p, phi, mpik, fk, deltafk, beta = 1000):
    return np.sum(0.5*p*p) + beta*0.5*np.sum((fk - fitfun(mpik, phi))*(fk - fitfun(mpik, phi))/(deltafk*deltafk))
    #return 0.5*p[0]**2 + 0.5*p[1]**2 + 0.5*p[2]**2 + beta*0.5*np.sum((fk - phi[0]-mpik*phi[1]-(mpik**2)*phi[2])**2 / (deltafk**2)) 

def fitfun(mpi, x):
    return x[0] + x[1]*mpi + x[2]*mpi*mpi

def del_H(p_0,phi_0,p2,phi2,mpik,fk,deltafk):
    H0 = ham(p_0,phi_0,mpik,fk,deltafk)
    #print(H0)
    H = ham(p2,phi2,mpik,fk,deltafk)
    #print(H)
    return (H-H0)/H0    


mpik = np.array([.176, .234, .260, .284, .324])
fk = np.array([960., 1025., 1055., 1085., 1130.])
deltafk = np.array([25., 20., 15., 10., 8.])

#fixed init values
p = np.array([1., 1., 1.])
phi = np.array([.9, .9, .9]) 

#p, phi = np.array([np.random.rand(), np.random.rand(), np.random.rand()]), np.array([np.random.rand(), np.random.rand(), np.random.rand()]) #random init values
Nmd = np.arange(1,101,1)
H_old = ham(p, phi, mpik, fk, deltafk) #energy of the init configuration
#print(H_old)
H_new = np.zeros(len(Nmd)) #array to store the energies of the final config
#dH = np.zeros(len(Nmd))
for i in range(len(Nmd)):
    #print(p,phi)
    p2, phi2 = leapfrog(p, phi, mpik, fk, deltafk, Nmd[i])
    H_new[i] = ham(p2, phi2, mpik, fk, deltafk)
    #print(p2,phi2)
    #dH[i] = np.absolute(del_H(p,phi,p2,phi2,mpik,fk,deltafk))
    

#print(p2, phi2)
xrange = np.arange(len(Nmd)) #+ 1
yval = np.abs((H_new - H_old)/H_old)

#plotting
plt.plot(xrange, yval, '.')
plt.yscale('log')
plt.xlabel('MD steps')
plt.ylabel('$\\vert\\frac{\\mathcal{H}[p_f, \\phi_f] - \\mathcal{H}[p_i, \\phi_i]}{\\mathcal{H}[p_i, \\phi_i]}\\vert$')
plt.title('Relative error vs. MD steps')
plt.grid()
plt.show()