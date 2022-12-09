import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

def initialstate(N):
    # Generates a random spin configuration for initial condition 
    state = 2*np.random.randint(2, size=(N,N))-1
    return state

N = 10

teststate = initialstate(N)

fftstate1 = fft(teststate)
fftstatefreq1 = fftfreq(N, 1/(N**2))
print(fftstatefreq1)
