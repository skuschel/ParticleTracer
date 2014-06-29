#!/usr/bin/env python2

import ParticleTracer as pt
import numpy as np


Bz = 0.05
def ode(*args):
    return pt.odediff(*args, B=lambda x,y,z,t:[0,0,Bz])

ekin = 100 * pt.const.keV
minit = pt.const.me
qinit = - pt.const.qe
xinit = [0,0,0, pt.ekin2p(ekin), 0, 0, qinit, minit]
print "xinit: " + str(xinit)

times = np.linspace(0,3e-9, 1000)
sol = pt.solve(ode, xinit, times)

#pt.plotsol(sol, times)

#Distmap
distmap = pt.distmap_plane(sol, [0,0,0], [1,0,0])
import matplotlib.pyplot as plt

plt.plot(times, distmap)
plt.xlabel('t')
plt.ylabel('distance to plane')
plt.show()
