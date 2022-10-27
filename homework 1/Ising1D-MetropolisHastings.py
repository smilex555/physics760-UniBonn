#Metropolis-MC-Simulation of 1-Dimensional Ising Model
import numpy as np
import matplotlib.pyplot as plt

# Use Boltzmann distribution as probability distribution 
def probability_dist(Hamiltonian, Temperature):
    return np.exp(-Hamiltonian / Temperature)

# calculate Hamiltonian of Ising model
# spins: configuration of spins per chain
def energy_ising(spins, h):
    energy=0
    for i in range(len(spins)):
        energy=energy+J*spins[i-1]*spins[i]
    energy= energy-h*sum(spins)
    return energy


def delta_energy_ising(spins,random_spin,N,h):
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
        
    # Using Metropolis-Hastings Algorithim
    data = []
    magnetization=[]
    energy=[]
    for i in range(MC_samples):
        #Each Monte Carlo step consists in N random spin moves
        for j in range(N):
            #Choosing a random spin
            random_spin=np.random.randint(0,N,size=(1))
            #Computing the change in energy of this spin flip
            delta=delta_energy_ising(spins,random_spin,N, field)

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
        #energy.append(energy_ising(spins,field))
    
    # calculate the average magnetization per spin after all samples
    average_magnetization = sum(magnetization)/MC_samples    
    # estimate std error 
    mag_std_err = np.std(magnetization, ddof=1)/np.sqrt(MC_samples)
 
    return  average_magnetization, mag_std_err
    


# setting important parameter
N = 20 # size of lattice 
MC_samples = 10000 #int(2**N) # number of samples / ensamble of possible spin configuration
T = 1 # "temperature" parameter
J = 1 # Strength of interaction between nearest neighbours
h = 0 # external field

# running MCMC

# compute magnatization for variable number of spins N for fixed external field h
N_L = np.arange(1,N+1) # array with all used N = 1,...,N_max
mag_N = [] #to save the average magnetization per spin for each N  
mag_N_err = []

for i in range(N):
    n = i+1
    #MC_samples = int(2**n)
    print(n, MC_samples)
    m = metropolis(N = n, MC_samples = MC_samples, Temperature = T, interaction = J, field = h)
    mag_N.append(m[0])
    mag_N_err.append(m[1])


# now variate the external field h for fixed number of spin N 
N = 10
MC_samples = 5000 # int(2**N)
num_h = 50   #quantity of h
mag_h = [] #to save average magnetization per spin for each field h
mag_h_err =[]
h_L = np.linspace(-1,1,num_h) # variation of h between -1 and 1

for i in h_L:
    #print(i)
    m = metropolis(N = N, MC_samples = MC_samples, Temperature = T, interaction = J, field = i)
    mag_h.append(m[0])
    mag_h_err.append(m[1])

#print(mag_h_err)


# analytical solution
def mag_exact(N,h):
    return T*np.sinh(h/T)*( (np.sqrt(np.sinh(h/T)**2 + np.exp(-4*J/T)) + np.cosh(h/T))**N - (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2 + np.exp(-4*J/T)))**N ) \
 /( np.sqrt(np.sinh(h/T)**2 + np.exp(-4*J/T))* ((np.sqrt(np.sinh(h/T)**2 + np.exp(-4*J/T)) + np.cosh(h/T))**N + (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2 + np.exp(-4*J/T)))**N ) ) 
               
# for fixed N and varation of h 
N = np.array([10, 700]) #considering different values for N
num_h = 50
h_exact = np.linspace(-1,1,num_h) # variation of h between -1 and 1
#mag_h_exact = []
mag2 = np.zeros([len(N), len(h_exact)])
for j in range(len(N)):
    for i in range(len(h_exact)):
        #print(i)
        #mag_anal = mag_exact(N, J, T, i)
        #print(mag_anal)
        #mag_h_exact.append(mag_anal)
        mag2[j, i] = mag_exact(N[j], h_exact[i])


# Plotting
plt.figure(figsize=(10,5))

#plot magnetization against N
plt.errorbar(N_L,mag_N,mag_N_err,ecolor='red', label='h=0')
plt.ylabel('magnetization m',fontdict={'size':10})
plt.xlabel('size of Lattice N',fontdict={'size':10})
plt.legend()
plt.show()

# mag against external field h with numerical and analytical methods
plt.errorbar(h_L,mag_h,mag_h_err,ecolor='red', label='N=10 (numerical)')
plt.plot(h_exact, mag2[0], label='N=10 (analytical)')
plt.plot(h_exact, mag2[1], label='N ->$ \infty $ (analytical)')
plt.ylabel('magnetization m',fontdict={'size':10})
plt.xlabel('external field h',fontdict={'size':10})
plt.legend(loc='upper left')
plt.show()
