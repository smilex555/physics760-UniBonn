#Metropolis-MC-Simulation of 2-Dimensional Ising Model
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from mpmath import *

# Use Boltzmann distribution as probability distribution 
def probability_dist(Hamiltonian, Temperature):
    return np.exp(-Hamiltonian / Temperature)

# calculate Hamiltonian of Ising model
# spins: configuration of spins per chain
'''
def energy_ising(J, spin_config, N, h):
    energy=0
    for i in range(len(spin_config)):
        for j in range(len(spin_config)):
            S = spin_config[i,j]
            nb = spin_config[(i+1)%N, j] + spin_config[i, (j+1)%N] + spin_config[(i-1)%N, j] + spin_config[i, (j-1)%N]
            energy += -J * nb * S - h*sum(map(sum, spin_config))
    # because we put h=0 for energy calculation the h-term is not neccesary
    #energy= energy-h*sum(map(sum, spin_config))
    return energy/4
'''
#this modified function doesn't need N as an argument, but still has it to work with legacy function calls
def energy_ising(J, spin_config, N, h):
    left = np.roll(spin_config, 1, 1)
    right = np.roll(spin_config, -1, 1)
    up = np.roll(spin_config, 1, 0)
    down = np.roll(spin_config, -1, 0)
    energy = (1/4)*-J*np.sum(spin_config*(left + right + up +down)) - h*np.sum(spin_config)
    return energy

def initialstate(N):
    # Generates a random spin configuration for initial condition 
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

#print(initialstate(10))
def mc_steps(spin_config, N, Temperature, J, h):
    # Using Metropolis-Hastings Algorithim      
    for i in range(N):
        #Each Monte Carlo step consists in N random spin moves
        for j in range(N):
            #Choosing a random spin
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = spin_config[a,b]
            #spin of the neighbors
            neighbors = spin_config[(i+1)%N, j] + spin_config[i, (j+1)%N] + spin_config[(i-1)%N, j] + spin_config[(i, (j-1)%N)] 
            #Computing the change in energy of this spin flip
            H_old = -J*neighbors*(s) - h*sum(map(sum, spin_config))
            spin_config[a,b] = -s             
            
            H_new = -J*neighbors*(-s) - h*sum(map(sum, spin_config))
            
            delta_E = H_new - H_old 
            spin_config[a,b] = s
            #delta_E = ( -J*neighbors*(-s) - h*sum(map(sum, spin_config)) ) - ( -J*neighbors*s - h*sum(map(sum, spin_config)) )  
                
            #Metropolis accept-rejection:
            if delta_E<0:
                # Accept the move if its negative
                # because new state is energetically favorable condition 
                s=-s
                #print('change')
            else:
                #If its positive, we compute the probability 
                probability=probability_dist(delta_E,Temperature)
                random=np.random.rand()
                # if the probability of boltzmann distribution is relatively high, accept change
                if random<=probability:
                    #Accept the move
                    s=-s
            spin_config[a,b] = s        

    return spin_config


def metropolis(N, MC_samples, eq_samples, T, J, h):
    magnetization=[]
    energy=[]  
    # intializing
    #Spin Configuration
    spin_config_ = initialstate(N)
    #print(spin_config)
    #just loop to update the lattice wthout storing the data until thermal equilibrium 
    for i in range(eq_samples):
        mc_steps(spin_config=spin_config_, N = N, Temperature = T, J=J, h=h)
    # now save the spin config of the updated lattice
    #print(spin_config)    
    for i in range(MC_samples):
        for _ in range(10):
            mc_steps(spin_config=spin_config_, N = N, Temperature = T, J=J, h=h)
        mc_steps(spin_config=spin_config_, N = N, Temperature = T, J=J, h=h)
        #Afer the MC step, we compute magnetization and energy per spin for a spin-configuration
        #magnetization.append(sum(map(sum,spin_config))/(N**2))
        magnetization.append(np.mean(spin_config_))
        energy.append(energy_ising(J, spin_config_,N, h)/(N**2))
        #energy.append() 
    # calculate the average magnetization per spin after all samples
    average_magnetization = sum(magnetization)/MC_samples 
    average_mag_abs = sum(np.absolute(magnetization))/MC_samples # average absolute magnatization per spin
    average_energy = sum(energy)/MC_samples   # calculate average energy per spin
    # estimate std error 
    mag_std_err = np.std(magnetization, ddof=1)/np.sqrt(MC_samples)
    abs_mag_err = np.std(np.absolute(magnetization), ddof=1)/np.sqrt(MC_samples)
    ener_std_err = np.std(energy, ddof=1)/np.sqrt(MC_samples)
    #print(spin_config)
    
    return  average_magnetization, mag_std_err, average_energy, ener_std_err, average_mag_abs, abs_mag_err
    

# setting important parameter
N = 5 # length of a quardratic lattice: N_x x N_y -> size of 2d-lattice 
eq_samples = 1000 # num of samples to reach thermal equilibrium 
MC_samples = 1000 #int(2**N) # number of samples / ensemble of possible spin configuration
T = 1 # "temperature" parameter
J = 0.2 # Strength of interaction between nearest neighbours
#h = 0 # external field


#metro = metropolis(N, MC_samples, eq_samples, T, J, h)
#print(metro)


# variate the external field h for fixed N and J
num_h = 10 #quantity of h
mag_h = [] #to save average magnetization per spin for each field h
mag_h_err =[]
h_L = np.linspace(-1,1,num_h) # variation of h between -1 and 1

for i in h_L:
    #print(i)
    m = metropolis(N, MC_samples, eq_samples, T, J, i)
    mag_h.append(m[0])
    mag_h_err.append(m[1])


#estimate average energy and magnetization as a function of J = [.25, 2]
num_J = 10 # quantitiy of J
av_energy = []
av_energy_err = []
abs_mag_J = []
abs_mag_J_err =[]
J_L = np.linspace(0.25,2,num_J)

for i in J_L:
    print(i)
    m = metropolis(N, MC_samples, eq_samples, T, i, 0)
    av_energy.append(m[2])
    av_energy_err.append(m[3])
    abs_mag_J.append(m[4])
    abs_mag_J_err.append(m[5])
    print(m)



#analytic solution
# critcal coupling J_c
J_c = (1/2) * np.log(1 + np.sqrt(2))

# abs magnetization per site with h=0 in thermodynamic limit
def abs_mag(J):
    if J > J_c:
        return (1 - (1/np.sinh(2*J)**4))**(1/8)
    else:
        return 0

def K(m):
    return special.ellipk(m)

# energy per site with h=0
def e(J,m):
    return - J * mp.coth(2*J) * ( 1 + (2/np.pi)*(2*np.tanh(2*J)**2 - 1)*K(m)*(4*mp.sech(2*J)**2 * np.tanh(2*J)**2) )   
    

#J_L = np.linspace(0.25,2,20) # variation of J between 0.25 and 2
energy_anal = np.zeros(len(J_L))
mag = np.zeros(len(J_L))
# loop over J and calculate energy and abs_magnetization
for i in range(len(J_L)):
    m = abs_mag(J_L[i]) 
    mag[i] = m
    energy_anal[i] = e(J_L[i],m)
         

#plotting
plt.figure(figsize=(10,5))


# mag against external field h with numerical and analytical methods
plt.errorbar(h_L,mag_h,mag_h_err,ecolor='red', label='N=5, J=0.2 (numerical)')
plt.ylabel('magnetization m',fontdict={'size':10})
plt.xlabel('external field h',fontdict={'size':10})
plt.legend(loc='upper left')
plt.show()


plt.errorbar(J_L[:],abs_mag_J[:],abs_mag_J_err[:],ecolor='red', label='N=10, h=0 (numerical)')
plt.plot(J_L, mag, label='N=10, h=0 (analytical)')
plt.ylabel('absolute magnetization m',fontdict={'size':10})
plt.xlabel('interaction J',fontdict={'size':10})
plt.legend(loc='upper left')
plt.show()

plt.errorbar(J_L,av_energy,av_energy_err,ecolor='red', label='N=10, h=0 (numerical)')
plt.plot(J_L, energy_anal,label = 'N=0, h=0 (analytical)')
plt.ylabel('energy',fontdict={'size':10})
plt.xlabel('interaction J',fontdict={'size':10})
plt.legend(loc='upper left')
plt.show()