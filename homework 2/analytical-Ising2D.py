import numpy as np
import matplotlib.pyplot as plt
from mpmath import *
import scipy as sp
from scipy import special

# critcal coupling J_c
J_c = (1/2) * np.log(1 + np.sqrt(2))

# abs magnetization per site with h=0 in thermodynamic limit
def abs_mag(J):
    if J > J_c:
        return (1 - (1/np.sinh(2*J)**4))**(1/8)
    else:
        return 0

def K(m):
    return special.ellipk(m)

# energy per site with h=0
def e(J,m):
    return - J * mp.coth(2*J) * ( 1 + (2/np.pi)*(2*np.tanh(2*J)**2 - 1)*K(m)*(4*mp.sech(2*J)**2 * np.tanh(2*J)**2) )   
    

J_len = np.linspace(0.25,2,50)
energy = np.zeros(len(J_len))
mag = np.zeros(len(J_len))
# loop over J and calculate energy and abs_magnetization
for i in range(len(J_len)):
    m = abs_mag(J_len[i]) 
    mag[i] = m
    energy[i] = e(J_len[i],m)
         
    
plt.figure()
plt.plot(J_len, energy)
plt.legend()
plt.xlabel('Interaction J')
plt.ylabel('Energy per site')
#plt.title('')
plt.show()

plt.plot(J_len, mag)
plt.legend()
plt.xlabel('Interaction J')
plt.ylabel('Energy per site')
plt.show()    
    
    