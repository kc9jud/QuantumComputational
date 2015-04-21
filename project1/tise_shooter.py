#!/usr/bin/env python3
# Filename: tise_shooting.py
# Author: Patrick Fasano/Ethan Sauer
# Created:  13-Jan-2015
# Modified: 03-Feb-2015
# Description: Solve the Schroedinger equation by the shooting method
#   using a Runge-Kutta ODE solver and the hybridized secant method.

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
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
    return 300*EV*(np.floor(x/BOHR+0.6) - np.floor(x/BOHR+0.4))

def error(E, plot=False, **kwargs):
    """ Calculates the error of the endpoint.
    
    Returns the error (final value of differential equation)
    
    E: energy of guess
    plot: add plot to queue
    **kwargs: keyword arguments for ode.solve()
    """

    if plot:
        return error_with_plot(E, **kwargs)
    else:
        endpoint = ode.solve(E = E, **kwargs)
        return endpoint[0]

def error_with_plot(E, **kwargs):
    """ Calculates the error of the endpoint and adds the intermediate plot to the
    plot queue.
    
    Returns the error (final value of differential equation)
    
    E: energy of guess
    **kwargs: keyword arguments for ode.solve()
    """
     
    xpoints,ypoints = ode.solve(return_points = True, E = E, **kwargs)
    plt.plot(xpoints,ypoints[0,:])
    return ypoints[0,-1]

####
# This method doesn't work. The integral has a bad habit of diverging at
# the end of the function.
#def normalize(E, V, interval):
#    """ Use the Romberg method of integration from the SciPy module to
#        normalize the wave function.
#    """
#    imin,imax,steps = interval
#    k = np.ceil(np.log2(steps - 1))
#    int_interval = (imin,imax,np.exp2(k)+1)
#    xpoints,ypoints = ode.solve(f_eq = schrod_eq,
#                        dep_i = (0,1),
#                        interval = int_interval,
#                        order = 4,
#                        return_points = True,
#                        V = V,
#                        E = E )
#    ypoints = np.power(ypoints[0,:],2) # strip out phi values and square
#    dx = abs((imax-imin))/(np.exp2(k))
#    print(dx,max(ypoints),int_interval)
#    int_val = integrate.romb(ypoints,dx)
#    print(int_val)
#    return 1/np.sqrt(int_val)

def normalize(E, V, interval):
    """ Use the Simpson method of integration from the SciPy module to
        normalize the wave function.
        
        Returns normalized numerical wavefunction (xpoints,ypoints).
    """
    xpoints,ypoints = ode.solve(f_eq = schrod_eq,
                        dep_i = (0,1),
                        interval = interval,
                        order = 4,
                        return_points = True,
                        V = V,
                        E = E )
    num_points = interval[2]
    ypoints_sq = np.power(ypoints[0,0:int(num_points/2)],2) # strip out phi values and square
    int_val = 2*integrate.simps(ypoints_sq,xpoints[0:int(num_points/2)])
    return np.array([xpoints,ypoints/np.sqrt(int_val)])

def solve_tise(V,x_interval,E_interval,mass,verbose=False):
    """ Solve the time-independent Schroedinger equation with hard walls and 
    arbitrary potential.
    
    This function uses the shooting method with a hybridized secant-method solver to 
    solve the Schroedinger equation for an arbitrary potential energy function.
    
    Returns tuple of energy and normalized numerical solution (xpoints,ypoints).
    
    V: potential energy function (takes one argument)
    x_interval: (xmin,xmax,steps) -- spatial interval in which energies should be found
    E_interval: (Emin,Emax) -- energy interval over which to search for states
    mass: mass of particle
    verbose: show output of secant solver
    """
    
    # work in unitless system: 0<=x<=1, potential in energy * (2m/hbar^2)
    xmin,xmax,nsteps = x_interval
    unitlessV = lambda x: V((x*abs(xmax-xmin)+xmin))*abs(xmax-xmin)**2*(2*mass/H_BAR**2)
    unitlessx_interval = (0,1,nsteps)
    unitlessE_interval = np.array(E_interval) * abs(xmax-xmin)**2 * (2*mass/H_BAR**2)
    
    # actually solve for the energy using the hybridized secant method
    unitless_energy = rootfind.hybrid_secant(f = error,
                        root_interval = unitlessE_interval,
                        tolerance = 1e-8,
                        verbose = verbose,
                          f_eq = schrod_eq,
                          dep_i = (0,1),
                          interval = unitlessx_interval,
                          order = 4,
                          V = unitlessV)

    xpoints,ypoints = normalize(unitless_energy,unitlessV,unitlessx_interval)
    
    # put our returned values back into their original units/scales
    num_soln = xpoints*abs(xmax-xmin)+xmin, ypoints[0,:]/np.sqrt(abs(xmax-xmin))
    energy = unitless_energy * abs(xmax-xmin)**-2 * H_BAR**2 / (2*mass)
    
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
    
#### uncomment this if you have trouble finding root intervals
#### it will plot the function of which we find roots using the secant method
#### caveat emptor -- it is very slow
##    error_points = functools.partial(error, 
##                 f_eq = schrod_eq,
##                 dep_i = (0,1),
##                 interval = (0,1,1000),
##                 order = 4,
##                 V = stepV)
    
##    energies = np.linspace(0,100,100)
##    error_points = list(map(error_points,energies))
##    print(len(energies),len(error_points))
##    plt.plot(energies,error_points)
##    plt.plot(energies,[0]*len(energies))
##    plt.show()
    
    print("\nInfinite square well:")
    plt.title("Infinite Square Well")
    energy,numer_soln = solve_tise(squarewellV,(0,BOHR,1000),(100*EV,150*EV),M_E)
    print("\tGround state energy:", energy/EV, "eV")
    plt.plot(*numer_soln)
    energy,numer_soln = solve_tise(squarewellV,(0,BOHR,1000),(400*EV,600*EV),M_E)
    print("\tFirst excited state energy:", energy/EV,"eV")
    plt.plot(*numer_soln)
    energy,numer_soln = solve_tise(squarewellV,(0,BOHR,1000),(950*EV,1300*EV),M_E)
    print("\tSecond excited state energy:", energy/EV,"eV")
    plt.plot(*numer_soln)
    
    plt.xlabel('x (m)')
    plt.ylabel('wave function $\psi(x)$')
    plt.xlim([0,BOHR])
    
    ax2 = plt.twinx()
    ax2.set_ylabel('potential (eV)')
    ax2.set_xlim([0,BOHR])
    ax2.set_ylim([0,500])
    potential_xpoints = np.linspace(0,BOHR,2000)
    potential_ypoints = np.array(list(map(squarewellV,potential_xpoints)))
    ax2.plot(potential_xpoints,potential_ypoints/EV,'k')
    plt.show()
    
    print("\nInfinite square well with symmetric step:")
    plt.title("Infinite Square Well w/ Central Step")
    energy,numer_soln = solve_tise(stepV,(0,BOHR,1000),(200*EV,400*EV),M_E)
    print("\tGround state energy:", energy/EV, "eV")
    plt.plot(*numer_soln)
    energy,numer_soln = solve_tise(stepV,(0,BOHR,1000),(500*EV,800*EV),M_E)
    print("\tFirst excited state energy:", energy/EV, "eV")
    plt.plot(*numer_soln)
    energy,numer_soln = solve_tise(stepV,(0,BOHR,1000),(1000*EV,1500*EV),M_E)
    print("\tSecond excited state energy:", energy/EV, "eV")
    plt.plot(*numer_soln)
    
    plt.xlabel('x (m)')
    plt.ylabel('wave function $\psi(x)$')
    plt.xlim([0,BOHR])
    
    ax2 = plt.twinx()
    ax2.set_ylabel('potential (eV)')
    ax2.set_xlim([0,BOHR])
    ax2.set_ylim([0,500])
    potential_xpoints = np.linspace(0,BOHR,2000)
    potential_ypoints = np.array(list(map(stepV,potential_xpoints)))
    ax2.plot(potential_xpoints,potential_ypoints/EV,'k')
    plt.show()
    
#Ex. 8.14 a) Code
    a = 1e-11 #m
    V0 = 50*EV #eV
    def V(x):
        return V0*x**2/(a**2)
    
    print("\n\nHarmonic Oscillator:\n")
    plt.title('Harmonic Oscillator')

    print("Solving for ground state:")
    ho_energy0,ho_soln0 = solve_tise(V,((-10*a), (10*a), 1000),(100*EV,200*EV),M_E)
    print("\tEnergy:", ho_energy0/EV, "eV")
    
    print("Solving for first excited state:")
    ho_energy1,ho_soln1 = solve_tise(V,((-10*a), (10*a), 1000),(400*EV,500*EV),M_E)
    print("\tEnergy:", ho_energy1/EV, "eV")
   
    print("Solving for second excited state:")
    ho_energy2,ho_soln2 = solve_tise(V,((-10*a), (10*a), 1000),(600*EV,700*EV),M_E)
    print("\tEnergy:",ho_energy2/EV, "eV")

    print("Energy difference between:")
    print("\tground state and first excited state:\t",(ho_energy1-ho_energy0)/EV,"eV")
    print("\tfirst and second excited states:\t",(ho_energy2-ho_energy1)/EV,"eV")

    plt.plot(*ho_soln0)
    plt.plot(*ho_soln1)
    plt.plot(*ho_soln2)
    
    plt.xlabel('x (m)')
    plt.ylabel('wave function $\psi(x)$')
    plt.xlim([-10*a,10*a])
    
    ax2 = plt.twinx()
    ax2.set_ylabel('potential (keV)')
    ax2.set_xlim([-10*a,10*a])
    ax2.set_ylim([0,5])
    potential_xpoints = np.linspace(-10*a,10*a,2000)
    potential_ypoints = np.array(list(map(V,potential_xpoints)))
    ax2.plot(potential_xpoints,potential_ypoints/(1000*EV),'k')
    plt.show()

#Ex. 8.14 b) Code
    a = 1e-11 #m
    V0 = 50*EV #eV
    def V(x):
        return V0*x**4/(a**4)
    
    print("\n\nAnharmonic Oscillator:\n")
    plt.title('Anharmonic Oscillator')

    print("Solving for ground state:")
    anho_energy0,anho_soln0 = solve_tise(V,((-10*a), (10*a), 1000),(100*EV,300*EV),M_E)
    print("\tEnergy:", anho_energy0/EV, "eV")
    
    print("Solving for first excited state:")
    anho_energy1,anho_soln1 = solve_tise(V,((-10*a), (10*a), 1000),(400*EV,800*EV),M_E)
    print("\tEnergy:", anho_energy1/EV, "eV")
    
    print("Solving for second excited state:")
    anho_energy2,anho_soln2 = solve_tise(V,((-10*a), (10*a), 1000),(1000*EV,2000*EV),M_E)
    print("\tEnergy:",anho_energy2/EV, "eV")

    print("Energy difference between:")
    print("\tground state and first excited state:\t",(anho_energy1-anho_energy0)/EV,"eV")
    print("\tfirst and second excited states:\t",(anho_energy2-anho_energy1)/EV,"eV")

    print("\nPlotting...")
    anho_energy0,anho_soln0 = solve_tise(V,((-5*a), (5*a), 1000),(100*EV,300*EV),M_E)
    anho_energy1,anho_soln1 = solve_tise(V,((-5*a), (5*a), 1000),(400*EV,800*EV),M_E)
    anho_energy2,anho_soln2 = solve_tise(V,((-5*a), (5*a), 1000),(1000*EV,2000*EV),M_E)

    plt.plot(*anho_soln0)
    plt.plot(*anho_soln1)
    plt.plot(*anho_soln2)
    plt.xlabel('x (m)')
    plt.ylabel('wave function $\psi(x)$')
    plt.xlim([-5*a,5*a])
    
    ax2 = plt.twinx()
    ax2.set_ylabel('potential (keV)')
    ax2.set_xlim([-5*a,5*a])
    ax2.set_ylim([0,50])
    potential_xpoints = np.linspace(-5*a,5*a,2000)
    potential_ypoints = np.array(list(map(V,potential_xpoints)))
    ax2.plot(potential_xpoints,potential_ypoints/(1000*EV),'k')
    plt.show()

