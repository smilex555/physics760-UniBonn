#libraries
import numpy as np
import itertools
import matplotlib.pyplot as plt

#parameters
J = 1
T = 1
ext = np.linspace(-1., 1., 50) #equally spaced values between -1 and 1 for the external magnetic field

#set total number of lattice points (N)
#N = 5 #fixed N - removed in final push
N = np.array([1, 5, 10, 20]) #considering 4 different values for N

#functions
def ham(spins, h = 0):
    """"Given all possible spin states,
    this function returns an array containing the correspoding energies"""
    n = len(spins[0])
    Ha = -J*np.sum(spins[:,:n-1]*spins[:,1:], axis=1) - h*np.sum(spins, axis=1)
    return Ha

def partition(hamiltonian):
    """Given an array of energy of all possible spin states,
    this function returns the partition function"""
    Z = np.sum(np.exp(-1*hamiltonian/T))
    return Z

#magnetisation vs external magnetic field for 4 different values of N
mag = np.zeros([len(N), len(ext)]) #create a zero matrix to store magnetisation values
for j in range(len(N)):
    #generate all possible spin states for the particular N
    spinstates = np.array([list(i) for i in itertools.product([-1, 1], repeat=N[j])])
    #calculate average spin
    avgspin = np.sum(spinstates, axis = 1)/N[j]
    
    #iterate over different values of the external field for a particular value of N
    for i in range(len(ext)):
        #calculate the energy of each state
        energy = ham(spinstates, ext[i]) #we have an array containing the energy corresponding to each state
        #calculate the probability of each state    
        probability = np.exp(-1*energy/T)/partition(energy)
        #sanity check below - removed in final push
        #print("Sanity Check: Total probability = ", np.sum(probability))
        mag[j, i] = np.sum(avgspin*probability) #every row contains the magnetisation values for a particular N
    
#plots
for i in range(len(N)):
    plt.plot(ext, mag[i], label=f'N={N[i]}')
    plt.legend()
plt.xlabel('Externel magnetic field')
plt.ylabel('Magnetisation')
plt.title('Magnetisation vs. External magnetic field')
plt.show()

#fig, ax = plt.subplots()
#ax.imshow(spinstates, interpolation='none', extent=[1,3,8,1]) 