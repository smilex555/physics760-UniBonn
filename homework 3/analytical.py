#libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

#function to calculate f given in the sheet
def fn(x, betajn = 1, betah = 1):
    return np.exp(0.5*betajn*x*x + betah*x)

#fucntion to calculate Z
def Z(betajn = 1, betah = 1, N = 20):
    zval = 0
    for i in range(N+1):
        zval += comb(N, i)*fn(N-2*i, betajn, betah)
    return zval

#function to calculate <beta*eps>
def betaeps(Z, betajn = 1, betah = 1, N = 20):
    betaepsval = 0
    for i in range(N+1):
        betaepsval += comb(N, i)*(0.5*betajn*(N-2*i)*(N-2*i) + betah*(N-2*i))*fn(N-2*i, betajn, betah)
    return -1/(N*Z) * betaepsval

#function to calculate <m>
def mag(Z, betajn = 1, betah = 1, N = 20):
    magval = 0
    for i in range(N+1):
        magval += comb(N, i)*(N-2*i)*fn(N-2*i, betajn, betah)
    return 1/(N*Z) * magval

#function calls
betajrange = np.linspace(0.2, 2, 100)
zval = Z(betajrange, 0.5)
betaepsval = betaeps(zval, betajrange, 0.5)
magval = mag(zval, betajrange, 0.5)

#beta-eps plot
plt.plot(betajrange, betaepsval)
plt.title('$\\langle\\beta\\epsilon\\rangle$ vs. $\\beta J$')
plt.xlabel('$\\beta J$')
plt.ylabel('$\\langle\\beta\\epsilon\\rangle$')
plt.show()

#m plot
plt.plot(betajrange, magval)
plt.title('$\\langle m \\rangle$ vs. $\\beta J$')
plt.xlabel('$\\beta J$')
plt.ylabel('$\\langle m \\rangle$')
plt.show()