#!/usr/bin/env python2

import ParticleTracer as pt
import numpy as np

#Example ExB drift
ekin = 0 * pt.const.keV #in x Direction
minit = pt.const.me
qinit = - pt.const.qe
xinit = [0,0,0, pt.ekin2p(ekin), 0, 0, qinit, minit]
print "xinit: " + str(xinit)
Bz = 0.1
Ex = 100
def ode(*args):
    return pt.odediff(*args, B=lambda x,y,z,t:[0,0,Bz], E=lambda x,y,z,t:[Ex,0,0])
times = np.linspace(0,3e-8, 10000)
sol = pt.solve(ode, xinit, times)

vExB = Ex * Bz / Bz**2
print "vExB : " + str(vExB)
print "pExB (nonrel): " + str(pt.const.me * vExB)

pt.plotsol(sol, times)
