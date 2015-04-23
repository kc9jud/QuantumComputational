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
    def __init__(self, V, E, a):
        pass
    
    def gen_rgrid(self):
        pass
    
    def solve(self):
        pass
    
    def solve_ode(self, l):
        pass
    
    def calc_phase_shift(self, l, points):
	#set up r1 and r2 using a
	#pick out X(r1) and X(r2)
	chi1 = points[r1, 1]
	chi2 = points[r2, 1]
	#find K
	K = r2*chi1/(r1*chi2)
	#Get correct Bessel functions jl and nl and plug in r values
	#scipy.special.jv(l,k*r)
	#scipy.special.yn(l,k*r)
	#tan^-1 of 3.19
	delta = numpy.arctan( (K*scipy.special.jv(l,k*r2)-scipy.special.jv(l,k*r1))/(K*scipy.special.yn(l,k*r2)-scipy.special.yn(l,k*r1)) )
        pass
    
    def f(self, theta):
        pass
    
    def diff_cross_sect(self, theta):
        pass
