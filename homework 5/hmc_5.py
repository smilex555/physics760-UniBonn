import numpy as np
import matplotlib.pyplot as plt
from leapfrog5 import leapfrog, ham
from tqdm import tqdm # for HMC progress bar

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
        random=np.random.rand()
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
        random=np.random.rand()
        #Metropolis accept-rejection:
        if random <= np.exp(delta_E):
            phi = phi_leap
            acceptence += 1
        else:
            phi = phi
        
        phi_all[i] = phi
    
    return phi_all, acceptence/N_samples


#define the autocorrelation function
def auto_corr_func(mc,mean,N,t):
    mc_sum = 0
    for i in range(N-t):
        mc_sum += (mc[i+1]-mean)*(mc[int(i+1+t)]-mean)    
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
N_config = 100000
N_burn = 10000

m_chain, accept_rate = HMC_alg(phi0, nmd, Tau, N_config, N_burn)

#####
plt.figure()
plt.plot(m_chain,'.', label='phi_2')
plt.xlabel('HMC tajectory')
plt.ylabel('O($\Phi$)')
plt.grid()
plt.legend()
plt.show()
#####


# put every autocorrelation steps together 
def auto_correlation(m_chain):
    # the estimated mean value of the simulated markov_chain
    est_mean = np.mean(m_chain)

    gamma_sum = 0 # the sum term of tau_int equation
    Gamma_t = [] # save the autocorrelation function for different t
    w = 0
    for t in range(len(m_chain)):
        g_t = Gamma_func(m_chain, est_mean, len(m_chain), t)
        gamma_sum += g_t 
        Gamma_t[t] = g_t 
        w+=1
        if g_t >= 0:
            break
    Gamma_0 = Gamma_func(m_chain, est_mean, len(m_chain), 0)    
    gamma_sum -= Gamma_0
    tau_int = 0.5*Gamma_0 + gamma_sum

    return Gamma_t, w, tau_int 

###
Gamma, W, tau_int = auto_correlation(m_chain)

markov_time = np.arange(1,W,1)

plt.figure()
plt.plot(markov_time, Gamma, '.', label='hmc')
plt.plot(markov_time, Gamma_func_2(markov_time,tau_int), '.', label='hmc')
plt.xlabel('markov chain time t')
plt.ylabel(' autocorrelation $\Gamma$(t)')
plt.label()
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
autocorr = []
for i in range([1,2,4,10,20]):
    mc_block = blocking(m_chain,i)
    autocorr[i] = auto_correlation(mc_block)
    plt.plot(autocorr[i], '.', label='bin width='+str(i))

plt.xlabel('t_mc / N_bin')
plt.ylabel(' autocorrelation $\Gamma$(t_mc)')
plt.xscale('log')
plt.yscale('log')
plt.label()
plt.show()
###


# bootstrap routine
def bootstrap(data):
    num_bs = 200
    mean_bs = np.zeros()

    for i in range(num_bs):
        N_config = len(data)
        
        # choose new random configuration of the markov chain
        bs_idx = np.random.randint(0,N_config, size=N_config)
        
        # new data configuration with the random drawn indeces 
        bs_data = data[bs_idx]
        
        # mean of the new data config
        mean_bs[i] = np.mean(bs_data)
        
    return np.mean(mean_bs), np.std(mean_bs)

###
bs_bin = [1,2,4,8,10,15,20,30] # used bin width
bs_mean = []
bs_std = []

for i in range(bs_bin):
    mc_block = blocking(m_chain,i)
    bs_mean[i], bs_std[i] = bootstrap(mc_block)
    
plt.figure()
plt.errorbar(bs_bin, bs_mean, bs_std, '.', label='estimated mean')
plt.xlabel('bin width')
plt.ylabel(' estimated mean of ')
plt.label()
plt.show()
### 


