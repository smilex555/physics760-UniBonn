import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# function to generate a random spin configuration
def initialstate(N):
    '''Generates a random spin configuration for initial condition '''
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

# parameters
N = 20
r = 0

# random initial state
teststate = initialstate(N)

# fft
fftstate = fft(teststate.flatten())
fftstatefreq = fftfreq(N*N)

# implementing the convolution
# this is done with keeping in mind how fft and fftfreq functions return the values
csum = fftstate[0]*fftstate[0]
for i in range(1, N*N):
    if(i != (N*N)/2):
        csum += fftstate[i]*fftstate[(N*N)-i]*np.exp(1j*fftstatefreq[i]*r)

# ignore the miniscule imaginary component that arises due to floating-point precision
Cr = np.real_if_close((1/(N*N*N*N))*csum)
print(f'{r}-correlator:', Cr)