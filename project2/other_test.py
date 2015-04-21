import ode
import numpy as np
import matplotlib.pylab as pylab

grid = np.linspace(1,10,100)

def g(x):
    return 1/x

res = ode.numerov(g,0.,1.,grid)

pylab.plot(grid,res)
#pylab.plot(grid,np.sin(grid))
#pylab.show()

res = ode.numerov(g,-1.62827,-0.319947,grid,direction='left')
pylab.plot(grid,res)
#pylab.plot(grid,np.sin(grid))
pylab.show()
