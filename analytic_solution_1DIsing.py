import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# define few important parameters
J = 1
T = 1
h = 0
N = 20

#h = Symbol('h')
#l_plus = np.exp(J/T) * (np.cosh(h/T) + np.sqrt(np.sinh(h/T)**2+np.exp(-4*J/T)))

#l_minus = np.exp(J/T) * (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2+np.exp(-4*J/T)))

#dl_p = l_plus.diff(h)
#dl_m = l_minus.diff(h)

#mag_anal = (T/N) * ( N*dl_p**(N-1) + N*dl_m**(N-1) )

#mag_anal1 = N*np.sinh(h/T)/(T*np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)))

#mag_anal2 = N*np.sinh(h/T)*( (np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)) + np.cosh(h/T))**N - (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)))**N ) \
# /( T*np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T))* ((np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)) + np.cosh(h/T))**N + (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)))**N ) ) 
               

def mag_exact(N,J,T,h):
    return N*np.sinh(h/T)*( (np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)) + np.cosh(h/T))**N - (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)))**N ) \
 /( T*np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T))* ((np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)) + np.cosh(h/T))**N + (np.cosh(h/T) - np.sqrt(np.sinh(h/T)**2 + np.exp(-J/T)))**N ) ) 
               

# for fixed h and variation of N
N_exact = np.arange(1,N+1) # array with all used N = 1,...,N_max
mag_N_exact = [] #to save the average magnetization per spin for each N  
for i in range(N):
    mag_anal = mag_exact(i, J, T, h)
    mag_N_exact.append(mag_anal)
    
#print(N_exact[4])
#print(mag_N_exact[4])    

# for fixed N and varation of h 
N = 1
num_h = 50
h_exact = np.linspace(-1,1,num_h) # variation of h between -1 and 1
mag_h_exact = []

for i in h_exact:
    #print(i)
    mag_anal = mag_exact(N, J, T, i)
    #print(mag_anal)
    mag_h_exact.append(mag_anal)

#print(h_exact)
#print(mag_h_exact)
#print(mag_anal)
   
# Plotting
plt.figure(figsize=(10,5))

plt.plot(N_exact, mag_N_exact)
plt.ylabel('magnetization m',fontdict={'size':20})
plt.xlabel('size of Lattice N',fontdict={'size':20})
plt.show()

plt.plot(h_exact, mag_h_exact)
plt.ylabel('magnetization m',fontdict={'size':20})
plt.xlabel('external field h',fontdict={'size':20})
plt.show()   