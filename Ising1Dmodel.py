#Metropolis-MC-Simulation of 1-Dimensional Ising Model
import numpy as np
import matplotlib.pyplot as plt

# Use Boltzmann distribution as probability distribution 
def probability_dist(Hamiltonian, Temperature):
    return np.exp(-Hamiltonian / Temperature)

# calculate Hamiltonian of Ising model
# spins: configuration of spins per chain
def energy_ising(spins):
    energy=0
    for i in range(len(spins)):
        energy=energy+J*spins[i-1]*spins[i]
    energy= energy-h*sum(spins)
    return energy


def delta_energy_ising(spins,random_spin):
    #If you do flip one random spin, the change in energy is:
    #(By using a reduced formula that only involves the spin
    # and its neighbours)
    if random_spin==N-1:
        PBC=0
    else:
        PBC=random_spin+1
        
    old = -J*spins[random_spin]*(spins[random_spin-1] + spins[PBC]) - h*spins[random_spin]
    new = J*spins[random_spin]*(spins[random_spin-1] + spins[PBC]) + h*spins[random_spin]
    
    return new-old


def metropolis(N=20, MC_samples=1000, Temperature = 1, interaction = 1, field = 0):
    
    # intializing
    #Spin Configuration
    spins = np.random.choice([-1,1],N)
        
    # Using Metropolis Algorithim
    data = []
    magnetization=[]
    energy=[]
    for i in range(MC_samples):
        #Each Monte Carlo step consists in N random spin moves
        for j in range(N):
            #Choosing a random spin
            random_spin=np.random.randint(0,N,size=(1))
            #Computing the change in energy of this spin flip
            delta=delta_energy_ising(spins,random_spin)

            #Metropolis accept-rejection:
            if delta<0:
                # Accept the move if its negative
                # because new state is energetically favorable condition 
                spins[random_spin]=-spins[random_spin]
                #print('change')
            else:
                #If its positive, we compute the probability 
                probability=probability_dist(delta,Temperature)
                random=np.random.rand()
                # if the probability of boltzmann distribution is relatively high, accept change
                if random<=probability:
                    #Accept the move
                    spins[random_spin]=-spins[random_spin]

        #data.append(list(spins))

        #Afer the MC step, we compute magnetization per spin and energy for a spin-configuration
        magnetization.append(sum(spins)/N)
        energy.append(energy_ising(spins))
    
    # calculate the average magnetization per spin after all samples
    aver_magnetization = sum(magnetization)/MC_samples
        
    return  aver_magnetization, energy
    


# setting important parameter
N = 15 # size of lattice
MC_samples = 1000 # number of samples
T = 1 # "temperature" parameter
J = 1 # Strength of interaction between nearest neighbours
h = 0 # external field

# running MCMC
data = metropolis(N = N, MC_samples = MC_samples, Temperature = T, interaction = J, field = h)

print("average_magnetization:",data[0])


'''
# Plotting
plt.figure(figsize=(20,10))

plt.subplot(2,1,2)
plt.plot(data[],'r')
plt.xlim((0,MC_samples))
plt.xticks([])
plt.yticks([])
plt.ylabel('Energy',fontdict={'size':20})
plt.xlabel('Time',fontdict={'size':20})
'''