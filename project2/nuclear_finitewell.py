#!/usr/bin/env python3

import radial_tise as rt
import numpy as np
import matplotlib.pyplot as plt

M_P = rt.M_P
MEV = rt.MEV
FM = rt.FM

#############
# Begin harmonic oscillator potential code
V0 = 23.2 * MEV  # J
R = 2.1 * FM  # m

E_MIN = -V0+1e-6*MEV
E_MAX = 1e-9*MEV


def step(x):
    """Return 1 if greater x>0, 0 if x<0."""
    return 0.5 * (np.sign(x) + 1)


def V_fw(r):
    """Return potential as a function of radius."""
    return -V0 * (1 - step(r - R))

solver = rt.RadialSolver(V_fw, (E_MIN, E_MAX), 0, 10*R, M_P/2, verbose=False, num_steps=1500)
# energy = solver.solve(tolerance=1e-3*MEV)
# print("E(0s)=", energy/MEV)
# print(solver.calculate_kink(energy))
# plt.plot(solver.rgrid/R, V_fw(solver.rgrid)/MEV, marker='.')
# plt.xlabel('r/a')
# plt.ylabel('$V(r)$')
# plt.show()

# # Plot kink
print("Plotting kink vs. E...")
Epoints = np.linspace(E_MIN, E_MAX, 500)
kpoints = []
for ep in Epoints:
    try:
        kpoints.append(solver.calculate_kink(ep))
    except:
        kpoints.append(0)

plt.xlabel('E')
plt.ylabel('$\psi^\prime_{right}(c) - \psi^\prime_{left}(c)$')
plt.plot(Epoints/MEV, kpoints, marker=',', label="l = 0")
plt.show()

# Actually solve and plot
energy = solver.solve(tolerance=1e-24*MEV)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0s')
#plt.plot(solver.rgrid/R, V_fw(solver.rgrid)/MEV, marker='.')
print("E(0s)=", energy/MEV)
print(solver.calculate_kink(energy))

plt.legend()
plt.title('Finite square well s-wave functions')
plt.xlabel('r/a')
plt.ylabel('$\psi(r)$')
plt.show()
