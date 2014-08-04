#!/usr/bin/env python2
'''
Example for using the paricle tracer for a calibration of a magnet spectrometer:

/\
|y x
-->
                         Detector plane
            o-----------------------------------------
            |            /
            |           /
 e-beam     |         /,
----------->s-----****
            |
    Bz=0    | Bz positive


Slit s at (0,0,0)
Origin o of plane coodinate system
'''

import ParticleTracer as pt
import numpy as np

#Magnetic Field:
Bz = 0.5
B = lambda x,y,z,t: [0,0,Bz * (x > 0)]

def ode(*args):
    return pt.odediff(*args, B=B)

ekin = np.linspace(10,300,20)  # unit: MeV
minit = pt.const.me
qinit = - pt.const.qe
xinit = [[-0.1,0,0, px, 0, 0, qinit, minit] for px in pt.ekin2p(ekin * pt.const.MeV)]
#print "xinit: " + str(xinit)

#Detector Plane:
planevec = [0,0.2,0]
planenormal = [0,1,0]
plane_ex = [1,0,0]

def trace(xinit):
    times = np.linspace(0,5e-9, 1000)
    sol = pt.solve(ode, xinit, times, hmax=2e-12)
    [solcross, angle] = pt.crossplane1st(sol, planevec, planenormal)
    crossplane = pt.transform2planecoords([solcross], planevec, planenormal, plane_ex)
    #pt.plotsol(sol, times)
    return [np.array(sol), np.array(crossplane[0])]

traces = np.array([trace(xi) for xi in xinit])

import matplotlib.pyplot as plt

#print traces.shape
#print (traces[:, 1]).shape

subplotsv = 1
subplotsh = 2

plt.subplot(subplotsv, subplotsh, 1)
for i in xrange(len(traces)):
    plt.plot(traces[i, 0][:,0], traces[i, 0][:,1], hold=True)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.ylim([0, planevec[1]])
    plt.xlim([-0.1, 1])

#print traces[:,1]
plt.subplot(subplotsv, subplotsh, 2)
crosses = traces[:,1]
xx = [c[0] for c in crosses]
plt.plot(xx, ekin)
plt.xlabel('x on detector plane [m]')
plt.ylabel('Ekin [MeV]')




plt.show(block=True)










