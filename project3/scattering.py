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
# H_BAR = 1.05457e-34 # J s
# M_E = 9.10938e-31 # kg
# M_P = 1.67262e-27 # kg
# BOHR = 5.29177e-11 # m
# EV = 1.602177e-19 # J
# #############################

##########CONSTANTS##########
H_BAR = 1 # 
M_E = 1 # 
M_P = 1836 # kg
BOHR = 1 # m
EV = 1/27.211 # hartree
#############################

class ScatteringSolver:
    xmin = 0 # starting position in r/a
    xmax = 10  # ending position in r/a
    xn   = 100 # number of steps
    
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
        self.rgrid,self.stepsize = self.a * np.linspace(self.xmin,
                                                        self.xmax,
                                                        self.xn+1,retstep=True)
    
    def k(self, r, l):
#        k = sqrt( (2*self.mass/H_BAR**2)(self.E - self.V(r)) )
        k = sqrt(self.schrod_eqn(r, l))
        return k
    
    def schrod_eqn(self, r, l):
        consts = 2*self.mass / H_BAR**2
        g = consts*(self.E - self.V(r)) - l*(l+1)/r**2
        return g
    
    def solve_ode(self, l):
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
        #set up r1 and r2 using a
        r2 = self.rgrid[-1]
        r1_index = -int(self.xn/10)
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

    def solve(self):
        l = 0
        rmax = self.a
        lmax = np.ceil(np.sqrt(self.ki)*rmax)
        temp_li = [] 
        while l<=lmax:
            points = self.solve_ode(l)
            shift = self.calc_phase_shift(l, points)
            temp_li.append(shift)
            l += 1
        self.phase_shifts = np.array(temp_li)
    
    def f(self, theta):
        retval = 0
        l=0
        for delta in self.phase_shifts:
            retval += (2*l+1)*np.exp(1j*delta)*np.sin(delta)*special.eval_legendre(l,np.cos(theta))
            l += 1
        return retval/self.ki
    
    def diff_cross_sect(self, theta):
        return np.abs(self.f(theta))**2
    
    def total_cross_sect(self):
        retval = 0
        l = 0
        for delta in self.phase_shifts:
            retval += (2*l+1)*(np.sin(delta))**2
            l += 1
        return 4*np.pi/self.ki**2 * retval

def V(r):
    return 50/(1+np.exp((r-.4)/.06))

solv = ScatteringSolver(V, 10, 1, 1)
solv.solve()
print(solv.phase_shifts)

solv_np = ScatteringSolver(np.zeros_like, 10, 1, 1)
for l in range(len(solv.phase_shifts)):
    points = solv.solve_ode(l)[1:]/solv.rgrid[1:]
    plt.plot(solv.rgrid[1:],points/np.max(points), marker='.')
    points = solv_np.solve_ode(l)[1:]/solv.rgrid[1:]
    plt.plot(solv.rgrid[1:],points/np.max(points), marker='.')
    plt.show()