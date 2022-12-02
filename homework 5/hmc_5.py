import numpy as np
import matplotlib.pyplot as plt
#from leapfrog5 import leapfrog, ham
from tqdm import tqdm # for HMC progress bar

def leapfrog(p, phi, nmd, tau = .1):
    eps = tau/nmd
    fpi, fphi = p, phi
    # first step
    fphi += .5*eps*fpi
    # intermediate steps
    for _ in range(nmd - 1):
        fpi += eps*(-2)*fphi
        fphi += eps*fpi
    # last step
    fpi -= eps*(-2)*fphi
    fphi += .5*eps*fpi
    return fpi, fphi

def ham(p, phi):
    return 0.5*np.power(p,2) + np.power(phi,2)


def HMC_alg(phi0,nmd,tau,N_samples,N_burn):
    #phi = 1
    acceptence = 0
    phi_all=np.zeros((N_samples))
    phi = phi0
    
    # burning some MC steps to get thermal equilibrium
    print('Burn-in')
    for i in tqdm(range(N_burn)):
        p0 = np.random.normal(0, 1)
        p_leap, phi_leap = leapfrog(p0, phi, nmd, tau)
        H_old = ham(p0, phi)
        H_new = ham(p_leap,phi_leap)
        delta_E = H_old - H_new
        random=np.random.uniform(0,1)
        #Metropolis accept-rejection:
        if random <= np.exp(delta_E):
            phi = phi_leap
        else:
            phi = phi
    
    print('HMC-core')
    for i in tqdm(range(N_samples)):
        p0 = np.random.normal(0, 1)
        p_leap, phi_leap = leapfrog(p0, phi, nmd, tau)
        H_old = ham(p0, phi)
        H_new = ham(p_leap,phi_leap)
        delta_E = H_old - H_new
        random=np.random.uniform(0,1)
        #Metropolis accept-rejection:
        if random <= np.exp(delta_E):
            phi = phi_leap
            acceptence += 1
        else:
            phi = phi
        
        phi_all[i] = phi
    
    return phi_all, acceptence/N_samples

# the observable function
def obs(phi):
    return np.cos(np.sqrt(1+phi**2))

#define the autocorrelation function
def auto_corr_func(mc,mean,N,t):
    mc_sum = 0
    for i in range(N-t):
        mc_sum += (mc[i]-mean)*(mc[int(i+t)]-mean)    
    return (1/(N-t)) * np.sum(mc_sum) 

#define the normalized autocorrelation function
def Gamma_func(mc,mean,N,t):
    c_t = auto_corr_func(mc,mean,N,t)
    c_0 = auto_corr_func(mc,mean,N,0)
    return c_t/c_0 #, c_t, c_0

#get autocorrelation function with autocorr_time
def Gamma_func_2(t,tau):
    return np.exp(-t/tau)
    

# integrated autocorrelation time
def integrated_time(gam_0, gam_t):
    return 0.5*gam_0 + gam_t

###

phi0 = 1
nmd = 3
Tau = .1
N_config = 1000000
N_burn = 10000

mc, accept_rate = HMC_alg(phi0, nmd, Tau, N_config, N_burn)
observ = obs(mc)
print(np.mean(observ))
print(accept_rate)

#####
plt.figure()
plt.plot(observ,'-')
plt.xlabel('HMC tajectory')
plt.ylabel('O($\Phi$)')
plt.grid()
plt.show()
#####


# put every autocorrelation steps together 
def auto_correlation(mc_data):
    # the estimated mean value of the simulated markov_chain
    est_mean = np.mean(mc_data)

    gamma_sum = 0 # the sum term of tau_int equation
    Gamma_t = [] # save the autocorrelation function for different t
    w = 0
    for t in tqdm(range(len(mc_data))):
        g_t = Gamma_func(mc_data, est_mean, len(mc_data), t)
        gamma_sum += g_t 
        Gamma_t.append(g_t) 
        w+=1
        if g_t <= 0:
            break
    
    Gamma_0 = Gamma_func(mc_data, est_mean, len(mc_data), 0)    
    gamma_sum -= Gamma_0
    tau_int = 0.5*Gamma_0 + gamma_sum

    return Gamma_t, w, tau_int 

###

Gamma, W, tau_int = auto_correlation(observ)

markov_time = np.arange(0,W,1)
print('W='+str(W), len(Gamma),len(markov_time))

plt.figure()
plt.plot(markov_time, Gamma, '.', label='gamma function')
plt.plot(markov_time, Gamma_func_2(markov_time,tau_int), '.', label='exp-function with tau_int')
plt.xlabel('markov chain time t')
plt.ylabel(' autocorrelation $\Gamma$(t)')
plt.legend()
plt.grid()
plt.show()
###


# Blocking / Binning the markov chain data
def blocking(mc_data, bin_width):
    if len(mc_data) % bin_width == 0:
        N_bin = len(mc_data) // bin_width
    else:
        N_bin = len(mc_data) // bin_width + 1  
    mean_block = np.zeros(N_bin)
    #idx = np.arange(0,N_bin,1)
    for j in range(N_bin):
        i = bin_width*j
        if j >= N_bin - 1:
            mean_block[j] = np.mean(mc_data[i:])
        else:
            mean_block[j] = np.mean(mc_data[i:i+bin_width])
    
    return mean_block    


###
# Demonstrate that the autocorrelation decreases as the bin width is increased
#autocorr = []
for i in [1,2,4,10,20]:
    mc_block = blocking(observ,i)
    #print(len(observ))
    print(len(mc_block))
    x = auto_correlation(mc_block)
    print(len(x[0]))
    #autocorr.append(x[0])
    plt.plot(x[0], '.', label='bin width='+str(i))

plt.xlabel('t_mc / N_bin')
plt.ylabel(' autocorrelation $\Gamma$(t_mc)')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show()
###



# bootstrap routine
def bootstrap(data,n_bs):
    mean_bs = np.zeros(n_bs)

    for i in range(n_bs):
        N_config = len(data)
        
        # choose new random configuration of the markov chain
        bs_idx = np.random.randint(0,N_config, size=N_config)
        
        # new data configuration with the random drawn indeces 
        bs_data = data[bs_idx]
        
        # mean of the new data config
        mean_bs[i] = np.mean(bs_data)
        
    return np.mean(mean_bs), np.std(mean_bs)

###
num_bs = 200 # number of bootstrap samples
bs_bin = [1,2,4,8,10,15,20,30] # used bin width
bs_mean = np.zeros(len(bs_bin))
bs_std = np.zeros(len(bs_bin))

for b_i,bin in enumerate(bs_bin):
    mc_block = blocking(observ,bin)
    bs_mean[b_i], bs_std[b_i] = bootstrap(mc_block, num_bs)

print('Mean and Std (Different Bin Widths:', bs_mean, bs_std)

#Behaviour of the function as a function of ensemble size --doesn't look right
"""
mc_block2 = blocking(observ, 20)
num_bs2 = np.arange(200, 211, 2)
bs_mean2 = np.zeros(len(num_bs2))
bs_std2 = np.zeros(len(num_bs2))

for num_i, num_bsloop in enumerate(num_bs2):
    bs_mean2[num_i], bs_mean2[num_i] = bootstrap(mc_block2, num_bsloop)
"""

plt.figure()
plt.errorbar(bs_bin, bs_mean, bs_std, fmt='.', capthick=1)
plt.xlabel('bin width')
plt.ylabel(' estimated mean $\mu$ ')
plt.show()

"""
plt.figure()
plt.errorbar(num_bs2, bs_mean2, bs_std2, fmt='.', capthick=1)
plt.xlabel('sample size')
plt.ylabel(' estimated mean $\mu$ ')
plt.show()
"""
### 