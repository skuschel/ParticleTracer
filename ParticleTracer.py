#!/usr/bin/env python2

"""
Particle Tracer capable of tracing a particle trajectory in a given E and B field configuration.

Author: Stephan Kuschel
"""


import numpy as np
import scipy.integrate as si


def fgamma(x, y, z, px, py, pz, q, m):
    return np.sqrt(1 + (px**2 + py**2 + pz**2) / (m**2 * 299792458.0**2))

def odediff((x, y, z, px, py, pz, q, m), t, E=lambda x,y,z,t:[0,0,0], B=lambda x,y,z,t:[0,0,0]):
    """
    Derivative of the x vector given to the ode solver, consting of
    [x, y, z, px, py, pz, charge, mass]
     0  1  2  3   4   5     6      7   
    E - Electric field (function of x,y,z)
    B - Magnetic field (function of x,y,z)
    """
    gm = fgamma(x, y, z, px, py, pz, q, m) * m
    (Ex, Ey, Ez) = E(x,y,z,t)
    (Bx, By, Bz) = B(x,y,z,t)
    
    ret = [px / gm, py / gm, pz / gm, 
        q * (Ex + py / gm * Bz - pz / gm * By),
        q * (Ey + pz / gm * Bx - px / gm * Bz),
        q * (Ez + px / gm * By - py / gm * Bx),
        0,0]
    return ret
    
    
#Beispiel: Elektron
keV = 1000 * 1.602 * 10**-19
Ekin = 10 * keV #Vorwaerts in x-Richtung
minit = 9.2 * 10**-31
qinit = -1.602 * 10**-19

xinit = [0,0,0, np.sqrt((Ekin + minit * 299792458.0**2) ** 2 / 299792458.0 ** 2 - minit**2 * 299792458.0 ** 2), 0, 0, qinit, minit]

print "xinit: " + str(xinit)


#Beispiel Lamour Radius stimmt!
Bz = 0.05
def odelamour(*args):
    return odediff(*args, B=lambda x,y,z,t:[0,0,Bz])
times = np.linspace(0,3e-9, 1000)
sol = si.odeint(odelamour, xinit, times)
lamour_num = (np.max(sol[:,0]) - np.min(sol[:,0])) / 2
gamma = 1 + Ekin / (minit * 299792458.0**2)
beta = np.sqrt(gamma**2 - 1) / gamma
lamour_ana = minit * beta * 299792458.0 / np.abs(qinit) / Bz 
print "lamour_numeric  (mm): " + str(lamour_num * 1000)
print "lamour_analytic (mm): " + str(lamour_ana * 1000) + " (non relativistic!)"



#Beispiel ExB drift
keV = 1000 * 1.602 * 10**-19
Ekin = 0 * keV #Vorwaerts in x-Richtung
minit = 9.2 * 10**-31
qinit = -1.602 * 10**-19
xinit = [0,0,0, np.sqrt((Ekin + minit * 299792458.0**2) ** 2 / 299792458.0 ** 2 - minit**2 * 299792458.0 ** 2), 0, 0, qinit, minit]
print "xinit: " + str(xinit)
Bz = 0.1
def odelamour(*args):
    return odediff(*args, B=lambda x,y,z,t:[0,0,Bz], E=lambda x,y,z,t:[100,0,0])
times = np.linspace(0,3e-8, 10000)
sol = si.odeint(odelamour, xinit, times)


#
Ekin = 0
minit = 9.2 * 10**-31
qinit = -1.602 * 10**-19
xinit = [1,0,0, np.sqrt((Ekin + minit * 299792458.0**2) ** 2 / 299792458.0 ** 2 - minit**2 * 299792458.0 ** 2), 0, 0, qinit, minit]
print "xinit: " + str(xinit)
def odeelectron(*args):
    return odediff(*args, E=lambda x,y,z,t:[x,y,0]*1/np.sqrt(x**2+y**2)**2*10, B=lambda x,y,z,t:[y,x,0]*1/np.sqrt(x**2+y**2)**2/80)
times = np.linspace(0,100e-8, 5700)
sol = si.odeint(odeelectron, xinit, times, hmax=1e-10)



import matplotlib.pyplot as plt

plt.subplot(3,2,1)
plt.plot(times, sol[:,0])
plt.xlabel("t")
plt.ylabel("x")

plt.subplot(3,2,3)
plt.plot(times, sol[:,1])
plt.xlabel("t")
plt.ylabel("y")

plt.subplot(3,2,5)
plt.plot(times, sol[:,2])
plt.xlabel("t")
plt.ylabel("z")

plt.subplot(3,2,2)
plt.plot(times, np.sqrt(sol[:,3]**2 + sol[:,4]**2 + sol[:,5]**2))
plt.xlabel("t")
plt.ylabel("P")

plt.subplot(3,2,4)
plt.plot(sol[:,0], sol[:,1])
plt.xlabel("x")
plt.ylabel("y")

plt.subplot(3,2,6)
plt.plot(sol[:,2], sol[:,0])
plt.xlabel("z")
plt.ylabel("x")

plt.show()



