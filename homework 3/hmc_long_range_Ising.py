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


def HMC_alg(phi0,J,h,N,nmd,N_samples,N_burn):
    #phi = 1
    acceptence = 0
    phi_all=np.zeros((N_samples))
    mu = 0
    sig = 1
    phi = phi0
    # burning some MC steps to get thermal equilibrium
    for i in range(N_burn):
        p0 = np.random.normal(mu, sig)
        p_leap, phi_leap = leapfrog(p0, phi, nmd, N, J, h)
        H_old = hamlf(p0, phi, N, J, h)
        H_new = hamlf(p_leap,phi_leap, N, J, h)
        delta_E = H_old - H_new
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random < np.exp(-delta_E):
            phi = phi_leap
        else:
            phi = phi
    
    for i in range(N_samples):
        p0 = np.random.normal(mu, sig)
        p_leap, phi_leap = leapfrog(p0, phi, nmd, N, J, h)
        H_old = hamlf(p0, phi, N, J, h)
        H_new = hamlf(p_leap,phi_leap, N, J, h)
        delta_E = H_new - H_old
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random < np.exp(-delta_E):
            phi = phi_leap
            acceptence += 1
        else:
            phi = phi
        
        phi_all[i] = phi
    
    return phi_all, acceptence/N_samples


def magnetization_obs(phi, h):
    beta = 1.
    return np.tanh(beta*h + phi)

def energy_obs(phi, N, h, J):
    beta = 1.
    return -(2*N*h*J*np.power(beta,2)*np.tanh(beta*h + phi) - J*beta + np.power(phi,2)) / (N*2*J*np.power(beta,2))
    
    #return -( np.power(phi,2)/(2*N*J*np.power(beta,2)) ) - h*np.tanh(beta*h + phi)


phi0 = 1 # starting initial phi
h = 0.5
J = np.arange(0.1,2.0,0.05)
N = np.array([5,10,15])   # number of sites 
nmd = 10
N_samples = 10000
N_burn = 1000


len_N = len(N)
len_J = len(J)
print(len_J)
accept_rate = np.zeros((len_N,len_J))
#phi_mc = np.zeros( len_N,len_J, N_samples)
energy = np.zeros( (len_N,len_J, N_samples))
magnet = np.zeros( (len_N,len_J, N_samples))


for m,n in enumerate(N):
    for i,j in enumerate(J):
        x = HMC_alg(phi0,j,h,n,nmd,N_samples,N_burn)
        phi_mc = x[0]
        accept_rate[m,i] = x[1]
        #print(n,j,x[1]) # show acceptence rate
        
        #if i == 0:
        #    print(phi_mc[:30])

        magnet[m,i,:] = magnetization_obs(phi_mc, h)
        energy[m,i,:] = energy_obs(phi_mc,n,h,j)
        #if i == 5:
        #    print(energy[m,i,:30])


# Use bootstrap method to calculate the errors

num_bs = 100
mag_bs = np.zeros((len_N, len_J, num_bs))
energy_bs = np.zeros((len_N, len_J, num_bs))

for b in range(num_bs):
    # choose new random configuration of the markov chain
    bs_config = np.random.randint(0,N_samples, size=N_samples)
    mag_bs[:,:,b] = np.mean(magnet[:,:,bs_config], axis = -1)
    energy_bs[:,:,b] = np.mean(energy[:,:,bs_config], axis= -1)

# get the average energy and mean magnetization per site and their errors of bootstrap
avg_mag = np.mean(mag_bs, axis= -1) 
avg_ener = np.mean(energy_bs, axis= -1)
avg_mag_err = np.std(mag_bs, axis= -1)
avg_ener_err = np.std(energy_bs, axis= -1)


fig = plt.figure(figsize=(10,7.5))

for n,n_val in enumerate(N):
    plt.errorbar(J, avg_mag[n,:], avg_mag_err[n,:], fmt='.', capthick=1, label='N='+str(n_val))

plt.xlabel('J')
plt.ylabel('mean magnetization <m>')
plt.legend()
plt.show()


for n,n_val in enumerate(N):
    plt.errorbar(J, avg_ener[n,:], avg_ener_err[n,:], fmt='.', capthick=1,label='N='+str(n_val))

plt.xlabel('J')
plt.ylabel('average energy <e>')
plt.legend()
plt.show()
