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

ekinx = np.linspace(10,300,20)  # unit: MeV
angle = np.linspace(-10,10,3) * np.pi / 180.0
minit = pt.const.me
qinit = - pt.const.qe
xinit = [[-0.1,0,0, p * np.cos(a), 0, p * np.sin(a), qinit, minit]
         for p in pt.ekin2p(ekinx * pt.const.MeV)
         for a in angle]
ekin = np.array([pt.p2ekin(np.sqrt(xi[3]**2 + xi[4]**2 + xi[5]**2))
                 for xi in xinit])
#print np.array(ekin) / pt.const.MeV
#print "xinit: " + str(xinit)

#Detector Plane:
planevec = [0,0.2,0]
planenormal = [0,1,0]
plane_ex = [1,0,0]

def trace(xinit):
    times = np.linspace(0,4e-9, 1000)
    sol = pt.solve(ode, xinit, times, hmax=2e-12)
    [solcross, angle] = pt.crossplane1st(sol, planevec, planenormal)
    crossplane = pt.transform2planecoords([solcross], planevec, planenormal, plane_ex)
    #pt.plotsol(sol, times)
    return [np.array(sol), np.array(crossplane[0])]

traces = np.array([trace(xi) for xi in xinit])

import matplotlib.pyplot as plt

#print traces.shape
#print (traces[:, 1]).shape

subplotsv = 2
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
plt.plot(xx, ekin/pt.const.MeV)
plt.xlabel('x on detector plane [m]')
plt.ylabel('Ekin [MeV]')

#show detector plane
plt.subplot(subplotsv, subplotsh, 3)
crosses = traces[:,1]
xx = [c[0] for c in crosses]
yy = [c[1] for c in crosses]
usept = ~np.isnan(xx)
ekins = np.compress(usept, ekin)
xx = np.compress(usept, xx)
yy = np.compress(usept, yy)

plt.scatter(xx, yy, c=ekins/pt.const.MeV)
plt.xlabel('x on detector plane [m]')
plt.ylabel('y on detector plane [m]')
cbar = plt.colorbar()
cbar.set_label('MeV')



plt.show(block=True)










