import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from mpmath import *
from tqdm import tqdm
from scipy.fft import fft, fftfreq

# Use Boltzmann distribution as probability distribution 
def probability_dist(Hamiltonian, Temperature):
    return np.exp(-Hamiltonian / Temperature)

# calculate Hamiltonian of Ising model
# spins: configuration of spins per chain
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

def mc_steps(spin_config, N, Temperature, J, h):
    # Using Metropolis-Hastings Algorithm      
    for i in range(N):
        #Each Monte Carlo step consists in N random spin moves
        for j in range(N):
            #Choosing a random spin
            a = np.random.randint(0, N)
            b = np.random.randint(0, N)
            s = spin_config[a,b]
            #spin of the neighbors
            neighbors = spin_config[(a+1)%N, b] + spin_config[a, (b+1)%N] + spin_config[(a-1)%N, b] + spin_config[(a, (b-1)%N)] 
            #Computing the change in energy of this spin flip
            # calculate old energy before and new energy after spin flip
            H_old = -J*neighbors*(s) - h*sum(map(sum, spin_config))
            spin_config[a,b] = -s  # flip the spin to calculate the whole energy           
            H_new = -J*neighbors*(-s) - h*sum(map(sum, spin_config))
            delta_E = H_new - H_old 
            spin_config[a,b] = s # flip the spin back after calculate delta_E
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
    # calculate the average magnetization per spin after all samples
    average_magnetization = sum(magnetization)/MC_samples 
    average_mag_abs = sum(np.absolute(magnetization))/MC_samples # average absolute magnatization per spin
    average_energy = sum(energy)/MC_samples   # calculate average energy per spin
    # estimate std error 
    mag_std_err = np.std(magnetization, ddof=1)/np.sqrt(MC_samples)
    abs_mag_err = np.std(np.absolute(magnetization), ddof=1)/np.sqrt(MC_samples)
    ener_std_err = np.std(energy, ddof=1)/np.sqrt(MC_samples)
    #print(spin_config)
    
    return  average_magnetization, mag_std_err, average_energy, ener_std_err, average_mag_abs, abs_mag_err, spin_config_

# setting important parameter
N = np.array([3, 5, 7, 9, 11]) # length of a quardratic lattice: N_x x N_y -> size of 2d-lattice 
eq_samples = 1000 # num of samples to reach thermal equilibrium 
MC_samples = 1000 # number of samples / ensemble of possible spin configuration
T = 1 # "temperature" parameter
J = 0.440686793509772 # Strength of interaction between nearest neighbours
#h = 0 # external field

#estimate average energy and magnetization as a function of J = [.25, 1]
num_J = 10 # quantitiy of J
av_energy = []
av_energy_err = []
abs_mag_J = []
abs_mag_J_err =[]


for i in N:
    print(i)
    m = metropolis(i, MC_samples, eq_samples, T, 0.1, 0)
    av_energy.append(m[2])
    av_energy_err.append(m[3])
    abs_mag_J.append(m[4])
    abs_mag_J_err.append(m[5])
    final_config = m[6]

    r = np.arange(i*i)
    Cr = np.zeros(len(r))
    for k in r:
        # fft
        fftstate = fft(final_config.flatten())
        fftstatefreq = fftfreq(i*i)

        # implementing the convolution
        # this is done with keeping in mind how fft and fftfreq functions return the values
        csum = fftstate[0]*fftstate[0]
        for j in range(1, i*i):
            if(j != (i*i)/2):
                csum += fftstate[j]*fftstate[(i*i)-j]*np.exp(1j*fftstatefreq[j]*k)

        # ignore the miniscule imaginary component that arises due to floating-point precision
        Cr[k] = np.real_if_close((1/(i*i*i*i))*csum)

    plt.plot(r, Cr)
    plt.title(f'$C_r$ vs. $r$ (N = {i})')
    plt.show()