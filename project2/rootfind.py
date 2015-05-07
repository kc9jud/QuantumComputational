#!/usr/bin/env python3
# Computational Physics
# Edward Kielb
# Spring 2014

""" rootfind.py -- library of rootfinding routines
     
    Language: Python 3
    Mark A. Caprio
    University of Notre Dame
    Written for Computational Methods in Physics, Spring 2014.
"""

import numpy
import matplotlib.pyplot

def bisection(f,interval,tolerance,verbose=False):
    """ Find root by bisection.

    The 'approximation' x_i at each iteration is defined by the
    midpoint of the interval.
    
    The 'error' x_i-x_(i-1) is defined by the change in midpoint from
    the midpoint of the last interval.  (Of course, for bisection,
    that is always half the width of the new interval.)

    Returns None if the sign of the function does not change on the
    given interval.  Otherwise, returns final midpoint x_i when
    termination condition is reached.

    f: function for rootfinding
    interval: tuple containing initial interval endpoints (xa,xb)
    tolerance: difference x_i-x_(i-1) at which search should terminate
    verbose (optional): whether or not to print iteration log
    """

    # set up initial bracketing interval
    #   Note: Sign of function *must* change in this interval for method to work.
    (xa,xb) = interval
    fxa = f(xa)
    fxb = f(xb)
    if (fxa*fxb >=0):
        # no sign change in interval
        return None

    # set up for first iteration
    xm = (xb + xa)/2
    error = (xb - xa)/2
    iteration_count = 0

    # bisect until tolerance reached
    while (abs(error) > tolerance):

        # increment iteration count
        iteration_count += 1
        
        # evaluate function
        fxa = f(xa)
        fxb = f(xb)
        fxm = f(xm)

        # find which subinterval contains root
        if (fxm == 0):
            # accidentally landed on root (often occurs for "toy" test intervals)
            xa = xm
            xb = xm
        elif ((fxa * fxm) < 0):
            # sign change is in left half of interval
            xb = xm
        else:
            # sign change is in right half of interval
            xa = xm

        # find new midpoint (and change in midpoint)
        xm_old = xm
        xm = (xb + xa)/2
        error = xm - xm_old

        # verbose iteration log
        if (verbose):
            print("iteration", iteration_count, "(bisection):",
                  "interval", (xa, xb), "root", xm)
            
    return xm

def newton(f,fp,x_guess,tolerance,verbose=False,max_iterations=100):
    """ Find root by Newton's method.

    The 'approximation' x_i at each iteration is defined by Newton's
    method, in terms of the previous approximation x_(i-1).
    
    The 'error' x_i-x_(i-1) is defined by the difference in successive
    approximations.

    Returns None if the maximum number of iterations is reached
    without satisfying the tolerance.  Also returns None if
    rootfinding lands on point where f has zero slope.  Otherwise,
    returns final approximation x_i when termination condition is
    reached.

    f: function for rootfinding
    fp: derivative of function for rootfinding (i.e., 'f prime')
    x_guess: initial guess point
    tolerance: error at which search should terminate
    verbose (optional): whether or not to print iteration log
    max_iterations (optional): limit on number of iterations
    """

    # set up for first iteration
    x = x_guess
    error = 2*tolerance  # set error big enough to get past while condition
    iteration_count = 0

    while iteration_count<max_iterations:
        newguess=x-f(x)/fp(x)
        newerror=newguess-x
        if verbose:
            print('Iteration: ',iteration_count+1,' Root Approximation: ',newguess,' Approximate Error: ',newerror)
        if abs(newerror)<error:
            return newguess
        x=newguess
        iteration_count+=1

    return None


def hybrid(f,fp,interval,tolerance,x_guess=None,verbose=False,max_iterations=100):
  """Find the root of a function using the hybridized Newton-Raphson method.
  
  This method enforces that a root fall within a certain interval, thus solving
  some of the stability problems of the Newton-Raphson method.
  """
  
  # set up our interval and iteration counter
  xmin = min(interval)
  xmax = max(interval)
  iter = 0
  
  # make an initial guess if we weren't given one
  # and check sanity of initial guess
  if x_guess == None:
    x = (xmax+xmin)/2
  if (x_guess < xmin) or (x_guess > xmax):
    raise ValueError("Initial guess is not within given interval")
  x = x_guess
  
  while (iter<max_iterations):
    iter += 1
    # Try Newton's method first
    x_guess = x - f(x)/fp(x)
    
    if (x_guess >= xmin) and (x_guess <= xmax):
      # Newton's method gave us a guess inside the interval
      newton = True
      error = abs(f(x)/fp(x))
    else:
      #Newton's method failed; fail back to bisection
      newton = False
      x_guess = (xmin+xmax)/2
      error = abs(xmax - x_guess)
    x = x_guess

    if ((f(x_guess)*f(xmin)) < 0):
      xmax = x_guess
    else:
      xmin = x_guess
    
    if verbose:
      print('Iteration:',iter,'Root Approximation:',x,'Newton:',newton,
            'Approximate Error:',error,'Interval:',xmin,xmax)
    
    if (error<tolerance):
      return x
  
  return None

def secant(f,x_guess,x_oldguess,tolerance,verbose=False,max_iterations=100):
    """ Find root by Newton's method with the secant approximation.

    The 'approximation' x_i at each iteration is defined by Newton's
    method, in terms of the previous approximation x_(i-1) and x_(i-2).
    
    The 'error' x_i-x_(i-1) is defined by the difference in successive
    approximations.

    Returns None if the maximum number of iterations is reached
    without satisfying the tolerance.  Also returns None if
    rootfinding lands on point where f has zero slope.  Otherwise,
    returns final approximation x_i when termination condition is
    reached.

    f: function for rootfinding
    x_guess: initial guess point
    x_oldguess: previous guess point
    tolerance: error at which search should terminate
    verbose (optional): whether or not to print iteration log
    max_iterations (optional): limit on number of iterations
    """

    # set up for first iteration
    x = x_guess
    oldguess = x_oldguess
    error = 2*tolerance  # set error big enough to get past while condition
    iteration_count = 0

    while iteration_count<max_iterations:
        newguess=x-f(x)*(x-oldguess)/(f(x) - f(oldguess))
        error=newguess-x
        if verbose:
            print('Iteration: ',iteration_count+1,' Root Approximation: ',newguess,' Approximate Error: ',error)
        if abs(error)<tolerance:
            return newguess
        oldguess=x
        x=newguess
        iteration_count+=1

    return None

def hybrid_secant(f, root_interval, tolerance=1e-10, x_guess=None, x_oldguess=None,
                  verbose=False, max_iterations=100, f_tolerance=10e-8, **kwargs):
    """ Find root by hybridized Newton's method with the secant approximation.
    
    The 'approximation' x_i at each iteration is defined by Newton's
    method, in terms of the previous approximation x_(i-1) and x_(i-2).
    If Newton's method gives a point outside the interval, the algorithm
    falls back on bisection.
    
    The 'error' x_i-x_(i-1) is defined by the difference in successive
    approximations.
    
    Returns None if the maximum number of iterations is reached
    without satisfying the tolerance.  Also returns None if
    rootfinding lands on point where f has zero slope.  Otherwise,
    returns final approximation x_i when termination condition is
    reached.
    
    f: function for rootfinding
    x_guess: initial guess point
    x_oldguess: previous guess point
    interval: bounds for valid roots
    tolerance: error at which search should terminate
    verbose (optional): whether or not to print iteration log
    max_iterations (optional): limit on number of iterations
    
    (c) Patrick Fasano, 2015
    """
    
    # make an initial guess if we weren't given one
    # and check sanity of initial guess
    (xmin,xmax) = root_interval
    if x_guess is None and x_oldguess is None:
        x_guess    = (xmax+xmin)/2 + (xmax-xmin)/10
        x_oldguess = (xmax+xmin)/2 - (xmax-xmin)/10
    if (x_guess < xmin) or (x_guess > xmax):
        raise ValueError("Initial guess is not within given interval")
    if (x_oldguess < xmin) or (x_oldguess > xmax):
        raise ValueError("Previous guess is not within given interval")    
    
    # set up for first iteration
    x = x_guess
    oldguess = x_oldguess
    error = 2*tolerance  # set error big enough to get past while condition
    iteration_count = 0
    
    f_xmin = f(xmin,     **kwargs)
    f_xmax = f(xmax,     **kwargs)
    f_oldguess = f(oldguess, **kwargs)
    f_x = f(x, **kwargs)
    
    while iteration_count<max_iterations:
        if (f_xmin*f_xmax > 0):
            raise ValueError("Even number of roots on interval ("+str(xmin)+","+str(xmax)+")")
        
        if verbose:
            print("Looking for roots in interval (",xmin,",",xmax,")")
        
        try:
            newguess=x-f_x*(x-oldguess)/(f_x - f_oldguess)
        except ZeroDivisionError:
            #Newton's method failed spectacularly; fail back to bisection
            newton = False
            newguess = (xmin+xmax)/2
            error = abs(xmax - newguess)
        
        if (newguess >= xmin) and (newguess <= xmax):
            # Newton's method gave us a guess inside the interval
            newton = True
            error=abs(newguess-x)
        else:
            #Newton's method failed; fail back to bisection
            newton = False
            newguess = (xmin+xmax)/2
            error = abs(xmax - newguess)
        
        
        if verbose:
            print('Iteration: ',iteration_count+1,' Root Approximation: ',newguess,' Approximate Error: ',error)
        if abs(error)<tolerance:# and abs(f_x)<tolerance:
            return newguess
#         elif abs(error)<tolerance and abs(f_x)>tolerance:
#             raise ValueError("Function appears to diverge to +/-inf at "+str(newguess))
        
        f_newguess = f(newguess, **kwargs)
        product = f_newguess*f_xmin
        if (product < 0):
            xmax = newguess
            f_xmax = f_newguess
        elif (product > 0):
            xmin = newguess
            f_xmin = f_newguess
        
        oldguess=x
        f_oldguess=f_x
        x=newguess
        f_x = f_newguess
        iteration_count+=1
    
    return None

def discrete(ypoints, nroot, verbose=False):
    """Finds the index of a root of a given number.
    
    Arguments:
       ypoints: 2D array of points in which to look for the root
       nroot: Number of the root we want
    Returns:
       i: Index of the root
    """

    i=0
    n=1   
    while(i < len(ypoints)-1):
        if(ypoints[i]*ypoints[i+1] < 0):
            if(n==nroot):
                return i
            else:
                n += 1
        i += 1

# test code
if (__name__ == "__main__"):

    # read in external libraries
    import math

    # define function for rootfinding
    def f_bench(x):
        return math.cos(x) - x
    def fp_bench(x):
        return -math.sin(x) - 1

#    func_values=[]
#    x_values=numpy.linspace(-10,10,200)
#    for xaxis in x_values:
#        func_values.append(f_bench(xaxis))
#    matplotlib.pyplot.plot(x_values,func_values,"b-")
#    matplotlib.pyplot.show()
    
    # bisection tests

    print("bisection(f_bench,(-1,0),1e-10,verbose=True)")
    print(bisection(f_bench,(-1,0),1e-10,verbose=True))

    print("bisection(f_bench,(0,1),1e-10,verbose=True)")
    print(bisection(f_bench,(0,1),1e-10,verbose=True))

    print("newton(f_bench,fp_bench,0.5,1e-10,verbose=True)")
    print(newton(f_bench,fp_bench,.5,1e-10,verbose=True))
    
    print("hybrid(f_bench,fp_bench,(0,1),1e-10,0.5,verbose=True)")
    print(hybrid(f_bench,fp_bench,(0,1),1e-10,0.5,verbose=True))
    
    print("hybrid(f_bench,fp_bench,(0,10),1e-10,5,verbose=True)")
    print(hybrid(f_bench,fp_bench,(0,10),1e-10,5,verbose=True))
    
    print("hybrid(f_bench,fp_bench,(0,15),1e-10,10,verbose=True)")
    print(hybrid(f_bench,fp_bench,(0,15),1e-10,10,verbose=True))
    
    print("secant(f_bench,0.45,0.55,1e-10,verbose=True)")
    print(secant(f_bench,0.45,0.55,1e-10,verbose=True))
