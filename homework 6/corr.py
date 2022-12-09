import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def initialstate(N):
    # Generates a random spin configuration for initial condition 
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

N = 20
r = 0

teststate = initialstate(N)

fftstate = fft(teststate.flatten())
fftstatefreq = fftfreq(N*N)

csum = fftstate[0]*fftstate[0]
for i in range(1, N*N):
    if(i != (N*N)/2):
        csum += fftstate[i]*fftstate[(N*N)-i]*np.exp(1j*fftstatefreq[i]*r)

Cr = np.real_if_close((1/(N*N*N*N))*csum)
print(Cr)