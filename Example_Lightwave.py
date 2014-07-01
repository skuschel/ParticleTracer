#!/usr/bin/env python2

import ParticleTracer as pt
import numpy as np

#Example Lightwave
ekin = 0 * pt.const.keV #in x Direction
minit = pt.const.me
qinit = - pt.const.qe
xinit = [0,0,0, pt.ekin2p(ekin), 0, 0, qinit, minit]
print "xinit: " + str(xinit)
#Lightwave
lambd = 1e-6 #Wavelength
a0 = 1
#---------------
omega = 2*np.pi*pt.const.c/lambd
E0 = a0 * omega * pt.const.me * pt.const.c / pt.const.qe
B0 = E0 / pt.const.c
k0 = omega / pt.const.c

print 'E0 (V/m): ' + str(E0)
print 'B0 (T): ' + str(B0)

def ode(*args):
    return pt.odediff(*args, B=lambda x,y,z,t:[0,B0*np.cos(-k0*z + omega*t),0], E=lambda x,y,z,t:[E0*np.cos(-k0*z + omega*t),0,0])
times = np.linspace(0,25e-15, 1000)
sol = pt.solve(ode, xinit, times, hmax=1e-18)


pt.plotsol(sol, times)
