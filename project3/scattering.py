#!/usr/bin/env python3
# Filename: scattering.py
# Author: Patrick Fasano/Ethan Sauer
# Created:  21-Apr-2015
# Modified: 21-Apr-2015
# Description: Calculate scattering phase shifts by
#   solving with a Numerov solver.

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

class ScatteringSolver:
    xmin = 0.5 # starting position in r/a
    xmax = 10  # ending position in r/a
    xn   = 100 # number of steps
    
    def __init__(self, V, E, a, mass, num_steps=None):
        self.V = V
        self.E = E
        self.a = a
        self.mass = mass
        
        if num_steps is not None:
            self.xn = num_steps
        
        self.ki = np.sqrt(2*m*E)/H_BAR
        
        self.gen_grids()
        
    
    def gen_grid(self):
        self.rgrid,self.stepsize = self.a * np.linspace(self.xmin,
                                                        self.xmax,
                                                        self.xn+1,retstep=True)
        
    
    def k(self, r):
        pass
    
    def schrod_eqn(self, r, E):
        consts = 2*self.mass / H_BAR**2
        g = consts*(E - self.V(r)) - self.l*(self.l+1)
        return g
    
    def solve_ode(self, l):
        pass
    
    def calc_phase_shift(self, l, points):
        pass
    
    def solve(self):
        pass
    
    def f(self, theta):
        pass
    
    def diff_cross_sect(self, theta):
        pass