#!/usr/bin/env python2

"""
Particle Tracer capable of tracing a particle trajectory in a given E and B field configuration.

Author: Stephan Kuschel
"""


import numpy as np

class const:
    c = 299792458.0     #lightspeed
    c2 = c ** 2
    me = 9.2 * 10**-31  #electron mass
    qe = 1.602 * 10**-19
    mp = 1836.2 * me    #proton mass
    keV = 1000 * qe


def ekin2p(ekin, m=const.me):
    return np.sqrt((ekin + m * const.c2) ** 2 / const.c2 - m**2 * const.c2)

def ekin2gamma(ekin, m=const.me):
    return 1 + ekin / (m * const.c2)


def _fgamma(x, y, z, px, py, pz, q, m):
    return np.sqrt(1 + (px**2 + py**2 + pz**2) / (m**2 * const.c**2))

def odediff((x, y, z, px, py, pz, q, m), t, E=lambda x,y,z,t:[0,0,0], B=lambda x,y,z,t:[0,0,0]):
    """
    Derivative of the x vector given to the ode solver, consting of
    [x, y, z, px, py, pz, charge, mass]
     0  1  2  3   4   5     6      7   
    E - Electric field (function of x,y,z)
    B - Magnetic field (function of x,y,z)
    """
    gm = _fgamma(x, y, z, px, py, pz, q, m) * m
    (Ex, Ey, Ez) = E(x,y,z,t)
    (Bx, By, Bz) = B(x,y,z,t)
    
    ret = [px / gm, py / gm, pz / gm, 
        q * (Ex + py / gm * Bz - pz / gm * By),
        q * (Ey + pz / gm * Bx - px / gm * Bz),
        q * (Ez + px / gm * By - py / gm * Bx),
        0,0]
    return ret


def simplesolve(ode, xinit=[0,0,0,0,0,0,const.qe,const.me]):
    import scipy.integrate as si
    times = np.linspace(0,3e-8, 10000)
    sol = si.odeint(ode, xinit, times)
    return sol
    
def solve(ode, xinit, times, **kwargs):
    import scipy.integrate as si
    return si.odeint(ode, xinit, times, **kwargs)

def distmap_plane(sol, planevec, planenormal):
    '''
    converts a list of coorinates into a list of distances to the given plane.
    sol         trajectory
    planevec    vector pointing to an aribitrary point inside the plane
    planenormal normal vector of the plane defined
    '''
    planenormal = np.array(planenormal)
    planevec = np.array(planevec)
    planenormal = planenormal / np.sqrt((planenormal*2).sum())
    origindist = np.dot(planevec, planenormal)
    ret = np.dot(sol[:,:3], planenormal) - origindist
    return ret

def crossing_zeros(distmap):
    '''
    Returns a list of indices and slopes, where the distmap array has crossd 0. These indices are floats because the zero crossing is linearly interpolated.
    '''
    index = []
    slope = []
    for i in xrange(len(distmap)-1):
         if distmap[i]*distmap[i+1] < 0:
            #linear interpolation
            slope.append(distmap[i+1] - distmap[i])
            index.append(i + np.abs(distmap[i] / slope[-1]))

    return index, slope



def plotsol(sol, times):
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


