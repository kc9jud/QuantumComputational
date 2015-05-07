#!/usr/bin/env python3
# Filename: radial_tise.py
# Author: Patrick Fasano/Ethan Sauer
# Created:  24-Feb-2015
# Modified: 30-Mar-2015
# Description: Solve the Schroedinger equation by the shooting method
#   using a Numerov ODE solver and the hybridized secant method.

import numpy as np
import scipy.integrate as integrate
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

class RadialSolver:
    xmin = -12
    xmax = 5
    
    
    def __init__(self, V, E_interval, l, a, mass, verbose=False, num_steps=200):
        if verbose:
            print("Initializing solver...")
        self.V = V
        self.Emin = E_interval[0]
        self.Emax = E_interval[1]
        self.a = a
        self.l = l
        self.mass = mass
        self.verbose = verbose
        
        if verbose:
            print("Generating grid...")
        self.gen_xgrid(num_steps)
        self.rgrid = self.r(self.xgrid)
        
#        if verbose:
#            print("Calculating initial turning point...")
#        self.calc_turnpoint_index((self.Emin+self.Emax)/2)
        max_kin = np.max(self.kinetic_energy(self.Emin, self.xgrid))
        if max_kin < 0:
            raise ValueError("Energy range classically forbidden:("+str(self.Emin)+","+str(self.Emax)+")")
        
        self.solution_points = None
        self.solution_energy = None
        
    
    def kinetic_energy(self, E, x):
        return E - (self.V(self.r(x)) + (H_BAR**2 * self.l * (self.l+1) / (2*self.mass*self.r(x)**2)))
    
#     def calc_turnpoint_index(self):
#         if self.verbose:
#             print("Calculating classical turning point")
#         
#         # wrap the function for use with rootfind; find the zero of potential energy
#         pot_en = lambda x: self.potential_energy(self.Emin, x)
# #        turnpoint = rootfind.hybrid_secant(f = pot_en, root_interval = ((self.xmin+self.xmax)/2,self.xmax), verbose=self.verbose)
#         turnpoint = rootfind.hybrid_secant(f = pot_en, root_interval = (self.xmin,self.xmax), verbose=self.verbose)
# #        turnpoint = 0
#         
#         # find the last x-point which is in the classically-allowed region
#         self.turnpoint_index = np.searchsorted(self.xgrid,turnpoint) - 1
#         
#         if self.verbose:
#             print("Classical turning point is at r =",self.r(self.xgrid[self.turnpoint_index]))
#             print("Classical turning point index is",self.turnpoint_index)
    
    def calc_turnpoint_index(self, E):
        if self.verbose:
            print("Calculating classical turning point for energy",E)
        
        kin_en_points = self.kinetic_energy(E, self.xgrid)
        if (self.l == 0):
            self.turnpoint_index = rootfind.discrete(kin_en_points, 1, verbose=self.verbose)
        else:
            self.turnpoint_index = rootfind.discrete(kin_en_points, 2, verbose=self.verbose)
        
        if self.verbose:
            print("Classical turning point index is",self.turnpoint_index)
            print("Classical turning point is at r =",self.rgrid[self.turnpoint_index])

    
    def zerosV(self, r):
        return np.zeros_like(x)
    
    def gen_xgrid(self, steps):
        self.xgrid,self.stepsize = np.linspace(self.xmin,self.xmax,steps+1,retstep=True)
    
    def r(self, x):
        return np.exp(x) * self.a
    
#     def Vx(self, x):
#         return self.V(self.r(x))
    
#    def calculate_deriv(self, points, direc):
#        pass
    
    def calculate_kink(self, E):
        self.calc_turnpoint_index(E)
        
        i = self.turnpoint_index
        h = self.stepsize
        
        points = self.solve_ode(E)
        
        #Coming in from right side
        dR = ( (-11./6)*points[i] + 3.*points[i+1] - (3./2)*points[i+2] + (1./3)*points[i+3] )/h
        
        #Coming in from left side
        dL = ( (11/6)*points[i] - 3*points[i-1] + (3/2)*points[i-2] - (1/3)*points[i-3] )/h
        
        kink = dR-dL
        if (self.verbose):
            print("dL =",dL,"dR =",dR,"Kink =",kink)
        return kink
    
    def schrod_eqn(self, x_grid, E):
        consts = 2*self.mass / H_BAR**2
        g = consts * (self.r(x_grid))**2 * (E - self.V(self.r(x_grid))) - (self.l+0.5)**2
        return g
    
    def solve_ode(self, E, direction=None, endpoint=None):
        if endpoint is None:
            endpoint = self.turnpoint_index
        
        if direction is None:
            # If a direction isn't specified, recurse downward and integrate
            # left and right sides.
            l_points = self.solve_ode(E, direction='right', endpoint=endpoint)
            r_points = self.solve_ode(E, direction='left', endpoint=endpoint)
            points = np.zeros_like(self.xgrid)
            points[:endpoint] = l_points[:endpoint] 
            points[endpoint:] = r_points[endpoint:] * (l_points[endpoint]/r_points[endpoint])
            
            points = self.normalize(points)
            return points
        
        
        if self.verbose:
            print("Calculating points...")
        
        points = ode.numerov(self.schrod_eqn, i0 = 0, i_slope = -1,
                                   x_grid = self.xgrid,
                                   direction = direction,
                                   end_index = endpoint,verbose=self.verbose, E=E)
        # Convert back from y(x) to chi(x)
        return points*np.sqrt(self.rgrid)
    
    def solve(self):
        energy = rootfind.hybrid_secant(f = self.calculate_kink,
                        root_interval = (self.Emin,self.Emax),
#                        tolerance = 1e-8,
                        verbose = self.verbose)
        
        self.solution_points = self.solve_ode(energy)
        self.solution_energy = energy
        
        return energy
    
    def normalize(self, points):
        intpoints = points**2 * self.rgrid
        intval = integrate.simps(intpoints,self.xgrid)
        return points/np.sqrt(intval)
    
if __name__ == "__main__":
    print("Begin...")
    
    def V(r):
#        return np.zeros_like(r)
        return -1/r
    
    solver = RadialSolver(V, (-.75,-0.4), 0, 1*BOHR, M_E, verbose=False)
    
    print("Plotting kink vs. E...")
    Epoints=np.linspace(-.75,-0.1,500)
    kpoints=[]
    for ep in Epoints:
        kpoints.append(solver.calculate_kink(ep))
    plt.plot(Epoints,kpoints)
    plt.show()
#    solver.verbose = True
    
    energy = solver.solve()
    print("E=",energy)
    
    left_points = solver.solve_ode(energy, direction='right')
    right_points = solver.solve_ode(energy, direction='left')
    
    right_points = right_points * (left_points[solver.turnpoint_index]/right_points[solver.turnpoint_index])
    
    plt.plot(solver.xgrid,left_points, marker='.')
    plt.plot(solver.xgrid,right_points, marker='.')
    plt.show()
    
    plt.plot(solver.rgrid,(left_points)**2, marker='.')
    plt.plot(solver.rgrid,(right_points)**2, marker='.')
    plt.show()
    
    
    
    
    solver = RadialSolver(V, (-.75,-0.2), 1, 1*BOHR, M_E)
    energy = solver.solve()
    print("E=",energy)
    
