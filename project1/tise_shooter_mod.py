#!/usr/bin/env python3
# Filename: tise_shooting.py
# Author: Patrick Fasano/Ethan Sauer
# Created:  13-Jan-2015
# Modified: 27-Jan-2015
# Description: Solve coupled first-order ordinary differential equations
#   by Runge-Kutta methods

import numpy as np
import scipy.integrate as integrate
from matplotlib import pylab
import functools

import ode
import rootfind

##########CONSTANTS##########
H_BAR = 1.05457e-34 # J s
M_E = 9.10938e-31 # kg
M_P = 1.67262e-27 # kg
BOHR = 5.29177e-11 # m
EV = 1.602177e-19 # J
###########################

def schrod_eq(r,x,V,E):
    """ Calculate derivatives for time-independent Schrödinger equation.
    
    The time-independent Schrödinger equation can be written as two first-order
    differential equations. This gives us the coupled differential equations:
    $d\psi/dx = \phi$
    $d\phi/dx = a[V(x)-E] \psi$
    
    Returns a vector of $d\psi/dx$, $d\phi/dx$.
    
    r: vector containing $\psi,\phi$
    """
    
    psi,phi = r
    return np.array([phi,(V(x)-E)*psi])

def squarewellV(x):
    return 0

def stepV(x):
    return 12e-17*(np.floor(x/BOHR+0.6) - np.floor(x/BOHR+0.4))

def V(x):
    V0 = 50*EV #eV
    return V0*x**2/(a**2)

def error(E, **kwargs):
    endpoint = ode.solve(E = E, **kwargs)
    return endpoint[0]

def error_plot(E):
    xpoints,ypoints = ode.solve(f_eq = schrod_eq,
                 dep_i = (0,1),
                 interval = (0,1,1000),
                 order = 4,
                 return_points = True,
                 V = stepV,
                 E = E )
    pylab.plot(xpoints,ypoints[0,:])
#    print(ypoints[:,-1])
    return ypoints[0,-1]

def normalize(E, V, interval):
    """ Use the Romberg method of integration from the SciPy module to
        normalize the wave function.
    """
    min,max,steps = interval
    k = np.ceil(np.log2(steps - 1))
    int_interval = (min,max,np.exp2(k)+1)
    xpoints,ypoints = ode.solve(f_eq = schrod_eq,
                        dep_i = (0,1),
                        interval = int_interval,
                        order = 4,
                        return_points = True,
                        V = V,
                        E = E )
    ypoints = np.power(ypoints[0,:],2) # strip out phi values and square
    dx = (max-min)/(np.exp2(k))
    A2 = integrate.romb(ypoints,dx)
    return 1/np.sqrt(A2)

def solve_tise(V,x_interval,E_interval,mass):
    xmin,xmax,nsteps = x_interval
    unitlessV = lambda x: V((x*abs(xmax-xmin)+xmin)) * abs(xmax-xmin)**2 * (2*mass/H_BAR**2)
    unitlessx_interval = (0,1,nsteps)
    unitlessE_interval = np.array(E_interval) * abs(xmax-xmin)**2 * (2*mass/H_BAR**2)
        
    unitless_energy = rootfind.hybrid_secant(f = error,
                        root_interval = unitlessE_interval,
                        tolerance = 1e-8,
                        verbose = True,
                          f_eq = schrod_eq,
                          dep_i = (0,1),
                          interval = unitlessx_interval,
                          order = 4,
                          V = unitlessV)
    energy = unitless_energy * abs(xmax-xmin)**-2 * H_BAR**2 / (2*mass)
    norm = normalize(energy,V,x_interval)
    xpoints,ypoints = ode.solve(f_eq = schrod_eq,
                        dep_i = (0,1),
                        interval = x_interval,
                        order = 4,
                        return_points = True,
                        V = V,
                        E = energy)
    num_soln = xpoints, ypoints[0,:]
    
    return (energy,num_soln)

if __name__ == "__main__":
    print("Begin...")
#     energy = rootfind.hybrid_secant(f = error,
#                root_interval = (0,20),
#                x_guess = 10,
#                x_oldguess = 12,
#                tolerance = 1e-8,
#                verbose = True,
#                  f_eq = schrod_eq,
#                  dep_i = (0,1),
#                  interval = (0,1,10000),
#                  order = 4,
#                  V = squarewellV)
#     
#     norm = normalize(energy,squarewellV,(0,1,4096))
#     
#     xpoints,ypoints = ode.solve(f_eq = schrod_eq,
#                         dep_i = (0,1),
#                         interval = (0,1,10000),
#                         order = 4,
#                         return_points = True,
#                         V = squarewellV,
#                         E = energy)
#     energy = energy * BOHR**-2 * H_BAR**2 / (2*M_E)

#    energy,numer_soln = solve_tise(squarewellV,(0,BOHR,8192),(1e-17,3e-17),M_E)
#    print(energy,"Joules")
#    pylab.plot(*numer_soln)
#    pylab.show()
    
#    energy,numer_soln = solve_tise(stepV,(0,BOHR,10000),(1e-17,8e-17),M_E)
#    print(energy, "Joules")
#    pylab.plot(*numer_soln)

#    pylab.plot(xpoints,stepV(xpoints)/100)
##    pylab.show()
    
    
##    error_points = functools.partial(error, 
##                 f_eq = schrod_eq,
##                 dep_i = (0,1),
##                 interval = (0,1,1000),
##                 order = 4,
##                 V = stepV)
    
##    energies = np.linspace(0,100,100)
##    error_points = list(map(error_points,energies))
##    print(len(energies),len(error_points))
##    pylab.plot(energies,error_points)
##    pylab.plot(energies,[0]*len(energies))
##    pylab.show()

#Ex. 8.14 Code
    a = 1e-11 #m
    energy,numer_soln = solve_tise(V,( (-10*a), (10*a), 10000),(100*EV,200*EV),M_E)
    print(energy, "Joules")
    pylab.plot(*numer_soln)
    energy,numer_soln = solve_tise(V,( (-10*a), (10*a), 10000),(400*EV,500*EV),M_E)
    print(energy, "Joules")
    pylab.plot(*numer_soln)
    energy,numer_soln = solve_tise(V,( (-10*a), (10*a), 10000),(600*EV,700*EV),M_E)
    print(energy, "Joules")
    pylab.plot(*numer_soln)
    pylab.show()
