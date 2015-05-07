#!/usr/bin/env python3
# Filename: scattering.py
# Author: Patrick Fasano/Ethan Sauer
# Created:  21-Apr-2015
# Modified: 21-Apr-2015
# Description: Calculate scattering phase shifts by
#   solving with a Numerov solver.

import numpy as np
import scipy.integrate as integrate
import scipy.special as special
import matplotlib.pyplot as plt
import functools

import ode
import rootfind

# ##########CONSTANTS##########
#H_BAR = 1.05457e-34 # J s
#M_E = 9.10938e-31 # kg
#M_P = 1.67262e-27 # kg
#BOHR = 5.29177e-11 # m
#EV = 1.602177e-19 # J
# #############################

##########CONSTANTS##########
H_BAR = 1 # 
M_E = 1 # 
M_P = 1836 # kg
BOHR = 1 # m
EV = 1/27.211 # hartree
ANGSTROM = 1.889726124*BOHR
#############################

class ScatteringSolver:
    xmin = 0 # starting position in r/a
    xmax = 10  # ending position in r/a
    xn   = 500 # number of steps
    
    def __init__(self, V, E, a, mass, num_steps=None, verbose=False):
        self.V = V
        self.E = E
        self.a = a
        self.mass = mass
        self.verbose = verbose
        
        if num_steps is not None:
            self.xn = num_steps
        
        self.ki = np.sqrt(2*mass*E)/H_BAR
        
        self.gen_grid()
    
    def gen_grid(self):
        wavelength = 2*np.pi/self.ki
        self.xmax = self.xmax*self.a
        # we need to integrate at least to four wavelengths
        self.xmax = max(self.xmax, 3*wavelength)
        # we want at least the default value for xn, but also at least 
        # 10 points per wavelength -- this matters for high-energy scattering
        self.xn = max(self.xn, int(np.ceil((10/wavelength) * (self.xmax-self.xmin))))
        # we need a minimum density of points, at least 20 points between x=0 and x=a
        self.xn = max(self.xn, int(20*self.xmax/self.a))
        self.rgrid,self.stepsize = np.array(np.linspace(self.xmin,
                                                        self.xmax,
                                                        self.xn+1,retstep=True))
    
    def k(self, r, l):
        """Finds the wave number, k, based on the radius and l value
        using the Schrodinger equation.

        Arguments:
           r: radius
           l: angular momentum number
        Returns:
           k: wave number
        """
#        k = sqrt( (2*self.mass/H_BAR**2)(self.E - self.V(r)) )
        k = sqrt(self.schrod_eqn(r, l))
        return k
    
    def schrod_eqn(self, r, l):
        """Defines the Schrodinger equation.

        Arguments:
           r: radius
           l: angular momentum number
        Returns:
           g: value of the Schrodinger equation
        """

        consts = 2*self.mass / H_BAR**2
        g = consts*(self.E - self.V(r)) - l*(l+1)/r**2
        return g
    
    def solve_ode(self, l):
        """Solves the Schrodinger equation for a given l value using the Numerov method
        in the ode module.

        Arguments:
           l: angular momentum number
        Returns:
           points: set of points containing the solution to the Schrodinger equation
        """

        if self.verbose:
            print("Calculating points...")
        
        # Set up initial conditions
        # We can't start at r=0, because the centrifugal term will diverge. Instead,
        # we use the central difference formula for the second derivative and X(0)=0
        # to write an (worse) approximation for the third point.
        points = np.zeros_like(self.rgrid)
        points[1] = 1/self.stepsize # This is an arbitrary -- it sets the normalization.
        points[2] = points[1] * (2 - self.schrod_eqn(self.rgrid[1],l)*(self.stepsize**2))

        points = ode.numerov(self.schrod_eqn, x_grid = self.rgrid, y_grid=points,
                             start_index=1, end_index = self.xn,
                             verbose=self.verbose, l=l)
        return points

    
    def calc_phase_shift(self, l, points):
        """Finds the phase shift of the wavelength using the method described in Gianozzi.
	
        Arguments:
           l: angular momentum number
           points: the array holding the solution to the Schrodinger equation
        Returns:
           delta: the phase shift
        """

	#set up r1 and r2 using a
        r2 = self.rgrid[-1]
        wavelength = 2*np.pi/self.ki
        r1_index = -int(2*wavelength/self.stepsize)
        r1 = self.rgrid[r1_index]
        
        #pick out X(r1) and X(r2)
        chi2 = points[-1]
        chi1 = points[r1_index]
        
        #find K
        K = (r2*chi1)/(r1*chi2)
        
        #Get correct Bessel functions jl and nl and plug in r values
        #special.sph_jn(l,k*r)
        #special.sph_yn(l,k*r)
        # These functions actually return a list of two arrays:
        #  - an array containing all jn up to l
        #  - an array containing all jn' up to l
        #tan^-1 of 3.19
        jn1 = special.sph_jn(l,self.ki*r1)[0][-1]
        yn1 = special.sph_yn(l,self.ki*r1)[0][-1]
        jn2 = special.sph_jn(l,self.ki*r2)[0][-1]
        yn2 = special.sph_yn(l,self.ki*r2)[0][-1]
        delta = np.arctan((K*jn2-jn1) / (K*yn2-yn1))
        return delta    

    def solve(self, lmax=None):
        """Finds the phase shifts for a number of different l values.
        Stores the resultant phase shifts in phase_shifts
        """

        l = 0
        rmax = self.a
        if lmax is None:
            lmax = 2*np.ceil(self.ki*rmax)
        temp_li = []
        while l<=lmax:
            points = self.solve_ode(l)
            shift = self.calc_phase_shift(l, points)
            temp_li.append(shift)
            l += 1
        self.phase_shifts = np.array(temp_li)
    
    def f(self, theta):
        """

        Arguments:
           theta:
        Returns:
           retval/ki:
        """

        retval = 0
        l=0
        for delta in self.phase_shifts:
            retval += (2*l+1)*np.exp(1j*delta)*np.sin(delta)*special.eval_legendre(l,np.cos(theta))
            l += 1
        return retval/self.ki
    
    def diff_cross_sect(self, theta):
        return np.abs(self.f(theta))**2
    
    def total_cross_sect(self):
        """Sums up the phase shifts in order to find the cross section.

        Returns:
           Cross section
        """

        retval = 0
        l = 0
        for delta in self.phase_shifts:
            retval += (2*l+1)*(np.sin(delta))**2
            l += 1
        return 4*np.pi/self.ki**2 * retval
    
    def plot_potential(self, **kwargs):
        """Plots the current potential data when called.

        Arguments:
           **kwargs: Any additional arguments to plot
        Returns:
           Displays the plot
        """

        plt.plot(self.rgrid, self.V(solv.rgrid), marker='.', **kwargs)
        plt.show()
    
    def plot_wave_functions(self, **kwargs):
        """Displays a plot of the wave functions. Nice for watching the phase shift decrease
        and the wavefunctions align.

        Arguments:
           **kwargs: Any additional plotting arguments
        Returns:
           Displays plot
        """

        rgrid = self.rgrid[1:]
        for l in range(len(self.phase_shifts)):
            points = self.solve_ode(l)[1:]/rgrid
            plt.plot(rgrid,points/np.max(points), marker='.', label="l="+str(l), **kwargs)
        plt.show()
    
    def plot_diff_cross_sect(self, **kwargs):
        """Displays a plot of the differential cross section.

        Arguments:
           **kwargs: Any additional arguments to plot
        Returns:
           Displays the plot
        """

        tgrid = np.linspace(-np.pi,np.pi,361)
        plt.polar(tgrid,self.diff_cross_sect(tgrid), marker='.', **kwargs)
        plt.show()

def V_WS(r):
    """Defines the Woods-Saxon potential function. Options are , Lennard-Jones, and
    Yukawa potentials.

    Arguments:
       r: radius
    Returns:
       Potential value
    """

    # Here's a nice Woods-Saxon potential
    return -50/(1+np.exp((r-.5)/.05))

eps = 5.9e-3*EV
sig = 3.57*ANGSTROM
def V_LJ(r):
    """Defines the Lennard-Jones potential function.
    
    Arguments:
       r: radius
    Returns:
       Potential value
    """

    #Here's a not nice Lennard-Jones potential
    return eps*( ((sig/r)**12) - 2*((sig/r)**6))

def V_Yuk(r):
    """Defines the Yukawa potential function.
    
    Arguments:
       r: radius
    Returns:
       Potential value
    """

    #Yukawa potential
    return -np.exp(-r)/r

#For Lennard-Jones
solv = ScatteringSolver(V_LJ, 1e-4*EV, .5*3.57*ANGSTROM, 1816)
solv.solve(lmax=6)
#For Woods-Saxon
#solv = ScatteringSolver(V_WS, 3, 1, 1)
#For Yukawa
#solv = ScatteringSolver(V_Yuk, 3, 0.25, 1)
#solv.solve()
solv.plot_potential()
solv.plot_wave_functions()
solv.plot_diff_cross_sect()
print(solv.phase_shifts)

#egrid = np.logspace(-5,3,750)
#Lennard-Jones
egrid = np.logspace(np.log10(1e-4*EV),np.log10(3e-3*EV),750)
points = []
for en in egrid:
    #For Lennard-Jones
    solv = ScatteringSolver(V_LJ, en, .5*3.57*ANGSTROM, 1816)
    #For Woods-Saxon
    #solv = ScatteringSolver(V_WS, en, 1, 1)
    #For Yukawa
    #solv = ScatteringSolver(V_Yuk, en, 0.5, 1)
    solv.solve(lmax=25)
    points.append(solv.total_cross_sect())
plt.plot(egrid/(1e-3*EV),points, marker='.')
plt.xscale('log')
plt.show()
