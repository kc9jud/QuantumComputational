import ode
import numpy as np
import matplotlib.pylab as pylab

grid = np.linspace(0,2*np.pi,100)

def g(x):
    return np.ones_like(x)

res = ode.numerov(g,0.,1.,grid,end_index=50)

pylab.plot(grid,res)
#pylab.plot(grid,np.sin(grid))
#pylab.show()

res = ode.numerov(g,0.,1.,grid,end_index=50,direction='left')
pylab.plot(grid,res)
#pylab.plot(grid,np.sin(grid))
pylab.show()
