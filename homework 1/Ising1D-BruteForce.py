#libraries
import numpy as np
import itertools
import matplotlib.pyplot as plt

#parameters
J = 1
T = 1
ext = np.linspace(-1., 1., 50)

#set total number of lattice points
N = 5

#functions
def ham(spins, h = 0):
    """"Given all possible spin states,
    this function returns an array containing the correspoding energies"""
    Ha = -J*np.sum(spins[:,:N-1]*spins[:,1:], axis=1) - h*np.sum(spins, axis=1)
    return Ha

def partition(hamiltonian):
    """Given an array of energy of all possible spin states,
    this function returns the partition function"""
    Z = np.sum(np.exp(-1*hamiltonian/T))
    return Z

#generate all possible spin states
spinstates = np.array([list(i) for i in itertools.product([-1, 1], repeat=N)])
avgspin = np.sum(spinstates, axis = 1)/N

mag = np.zeros(len(ext))
for i in range(len(mag)):
    #calculate the energy of each state
    energy = ham(spinstates, ext[i]) #we have an array containing the energy corresponding to each state
    #calculate the probability of each state    
    probability = np.exp(-1*energy/T)/partition(energy)
    #sanity check below - removed in final push
    #print("Sanity Check: Total probability = ", np.sum(probability))
    mag[i] = np.sum(avgspin*probability)

#plots
plt.plot(ext, mag)
plt.show()

#fig, ax = plt.subplots()
#ax.imshow(spinstates, interpolation='none', extent=[1,3,8,1])