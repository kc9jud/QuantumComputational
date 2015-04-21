#!/usr/bin/env python3
# Filename: ode.py
# Author: Patrick Fasano/Emmy Kunce/Ethan Sauer
# Created:  4-7-2014
# Modified: 3-2-2015
# Description: Solve coupled first-order ordinary differential equations
#   by Runge-Kutta methods

import numpy as np

def solve(f_eq,dep_i,interval,order=1,return_points=False, **kwargs):
  indep_start = interval[0]
  indep_stop = interval[1]
  num_steps = interval[2]
  h = (indep_stop - indep_start)/num_steps
  # assume an equation of the form dx/dt = f(x,t)
  r = dep_i[:]
  
  tpoints = np.linspace(*interval)
  if return_points:
    rpoints = np.zeros((len(r),len(tpoints)))

  for i in range(len(tpoints)):
    # step through the tpoints, assign the rpoints if we want to save them
    if return_points:
      rpoints[:,i] = r[:]

    if (order == 1):
      k1 = h * f_eq(r,tpoints[i],**kwargs)
      r = r + k1
    elif(order == 2):
      k1 = h * f_eq(r,tpoints[i],**kwargs)
      k2 = h * f_eq(r + (0.5*k1), tpoints[i]+(0.5*h),**kwargs)
      r = r + k2
    elif(order == 4):
      k1 = h * f_eq(r, tpoints[i],**kwargs)
      k2 = h * f_eq(r+(0.5*k1), tpoints[i]+(0.5*h),**kwargs)
      k3 = h * f_eq(r+(0.5*k2), tpoints[i]+(0.5*h),**kwargs)
      k4 = h * f_eq(r+k3, tpoints[i]+h,**kwargs)
      r = r + (k1 + 2*k2 + 2*k3 +k4)/6
  
  if return_points:
    return (tpoints,rpoints)
  else:
    return r

if __name__ == "__main__":
  from matplotlib import pylab
  from math import sin
  def plot_approx(f_eq, dep_i, interval, order=1, exact_soln=None, **kwargs):
    pylab.xlabel("t")
    pylab.ylabel("x(t)")
    xpoints,ypoints = solve(f_eq,dep_i,interval,order,return_points=True)
    label = "order "+str(order)+", "+str(interval[2])+"pts"
    pylab.plot(xpoints,ypoints[0,:], label=label)
    
    if(exact_soln != None):
      yexact = list(map(exact_soln,xpoints))
      pylab.plot(xpoints,yexact, label="exact soln")

  def f(r,t):
    return np.array([sin(t) - r[0]**3])
  def g(x,t):
    return (-2*t*x)/(1+t**2)
  def g_exact(t):
    return -25/(t**2 +1)
    
  plot_approx(f, [0], (0.,10.,50), order=1)
  plot_approx(f, [0], (0.,10.,30), order=2)
  plot_approx(f, [0], (0.,10.,30), order=4)
  pylab.legend()
  pylab.show()
