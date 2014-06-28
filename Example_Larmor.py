#!/usr/bin/env python2

import ParticleTracer as pt
import numpy as np

#Larmor Radius is correct
Bz = 0.05
def odelarmor(*args):
    return pt.odediff(*args, B=lambda x,y,z,t:[0,0,Bz])

ekin = 10 * pt.const.keV
minit = pt.const.me
qinit = - pt.const.qe
xinit = [0,0,0, pt.ekin2p(ekin), 0, 0, qinit, minit]
print "xinit: " + str(xinit)

times = np.linspace(0,3e-8, 1000)
sol = pt.solve(odelarmor, xinit, times)
larmor_num = (np.max(sol[:,0]) - np.min(sol[:,0])) / 2
gamma = pt.ekin2gamma(ekin)
beta = np.sqrt(gamma**2 - 1) / gamma
larmor_ana = minit * beta * 299792458.0 / np.abs(qinit) / Bz 
print "Larmor_numeric  (mm): " + str(larmor_num * 1000)
print "Larmor_analytic (mm): " + str(larmor_ana * 1000) + " (non relativistic!)"

pt.plotsol(sol, times)

