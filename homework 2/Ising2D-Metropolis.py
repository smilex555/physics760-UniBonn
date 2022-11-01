#Metropolis-MC-Simulation of 2-Dimensional Ising Model
import numpy as np
import matplotlib.pyplot as plt

# Use Boltzmann distribution as probability distribution 
def probability_dist(Hamiltonian, Temperature):
    return np.exp(-Hamiltonian / Temperature)

# calculate Hamiltonian of Ising model
# spins: configuration of spins per chain
def energy_ising(J,spin_config, N, h):
    energy=0
    for i in range(len(spin_config)):
        for j in range(len(spin_config)):
            S = spin_config[i,j]
            nb = spin_config[(i+1)%N, j] + spin_config[i, (j+1)%N] + spin_config[(i-1)%N, j] + spin_config[i, (j-1)%N]
            energy += -nb * S * J 
    energy= energy-h*sum(map(sum, spin_config))
    return energy


def initialstate(N):
    # Generates a random spin configuration for initial condition 
    state = 2*np.random.randint(2, size=(N,N))-1
    return state


def metropolis(N,MC_samples, Temperature, interaction, field):
    
    # intializing
    #Spin Configuration
    spin_config = initialstate(N)
         
    # Using Metropolis-Hastings Algorithim
    magnetization=[]
    energy=[]
    for m in range(MC_samples):     
        for i in range(N):
            #Each Monte Carlo step consists in N random spin moves
            for j in range(N):
                #Choosing a random spin
                a = np.random.randint(0, N)
                b = np.random.randint(0, N)
                s = spin_config[a,b]
            
                neighbors = spin_config[(i+1)%N, j] + spin_config[i, (j+1)%N] + spin_config[(i-1)%N, j] + spin_config[(i, (j-1)%N)] 
                #Computing the change in energy of this spin flip
                delta_E = 2 * neighbors * s
                
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


        #Afer the MC step, we compute magnetization per spin and energy for a spin-configuration
        magnetization.append(sum(map(sum,spin_config))/(N**2))
        energy.append(energy_ising(interaction, spin_config, N, field)/(N**2))
    
    # calculate the average magnetization per spin after all samples
    average_magnetization = sum(magnetization)/MC_samples 
    average_energy = sum(energy)/MC_samples   
    # estimate std error 
    mag_std_err = np.std(magnetization, ddof=1)/np.sqrt(MC_samples)
 
    return  average_magnetization, mag_std_err, average_energy
    

# setting important parameter
N = 10 # length of a quardratic lattice: N x N -> size of 2d-lattice 
MC_samples = 10000 #int(2**N) # number of samples / ensamble of possible spin configuration
T = 1 # "temperature" parameter
J = 0.3 # Strength of interaction between nearest neighbours
h = 0 # external field


metro = metropolis(N = N, MC_samples = MC_samples, Temperature = T, interaction= J, field = h)

print(metro)