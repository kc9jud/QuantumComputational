#!/usr/bin/env python3

import radial_tise as rt
import numpy as np
import matplotlib.pyplot as plt

M_P = 1.6726219e-27 # kg
MEV = 1.6021766e-13 # J
R = 1.27 * 16**(1/3) * 1e-15 # m

# #############
# # Begin Woods-Saxon potential code
# a = 0.67 * 1e-15 # m
#
# def V_WS(r):
#     return -51*MEV/(1+np.exp((r-R)/a))
#
# # Plot Woods-Saxon potential
# solver = rt.RadialSolver(V_WS, (-15*MEV,-2*MEV), 2, R/4, M_P, verbose=False, num_steps=1000)
# plt.plot(solver.rgrid/R, V_WS(solver.rgrid), marker='.')
# plt.xlabel('r/a')
# plt.ylabel('$V(r)$')
# plt.show()
#
# # Plot kink
# print("Plotting kink vs. E...")
# Epoints=np.linspace(-15*MEV,-2*MEV,500)
# kpoints=[]
# for ep in Epoints:
#     kpoints.append(solver.calculate_kink(ep))
# plt.xlabel('E')
# plt.ylabel('$\psi^\prime_{right}(c) - \psi^\prime_{left}(c)$')
# plt.plot(Epoints/MEV,kpoints, marker='.')
# plt.show()
# print(solver.solve()/MEV)
#
# # Actually solve and plot
# solver = rt.RadialSolver(V_WS, (-50*MEV,-20*MEV), 0, R/4, M_P, verbose=False, num_steps=1000)
# energy = solver.solve()
# plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0s')
# print("E(0s)=",energy/MEV)
# solver = rt.RadialSolver(V_WS, (-20*MEV,-14*MEV), 1, R/4, M_P, verbose=False, num_steps=1000)
# energy = solver.solve()
# plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0p')
# print("E(0p)=",energy/MEV)
# solver = rt.RadialSolver(V_WS, (-6*MEV,-2*MEV), 2, R/4, M_P, verbose=False, num_steps=1000)
# energy = solver.solve()
# plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0d')
# print("E(0d)=",energy/MEV)
# solver = rt.RadialSolver(V_WS, (-6*MEV,-2*MEV), 0, R/4, M_P, verbose=False, num_steps=1000)
# energy = solver.solve()
# plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='1s')
# print("E(1s)=",energy/MEV)
# plt.legend()
# plt.title('Woods-Saxon s-wave functions')
# plt.xlabel('r/a')
# plt.ylabel('$\psi(r)$')
# plt.show()




#############
# Begin harmonic oscillator potential code
omega = 6.53*MEV/rt.H_BAR # m
V1 = 48.6 * MEV # J
R = 1.27 * 136**(1/3) * 1e-15 # m

def V_HO(r):
    return -V1 + (1/2) * M_P * omega**2 * r**2

# # Plot Woods-Saxon potential
solver0 = rt.RadialSolver(V_HO, (-V1+.01,-20*MEV), 0, 0.025*R, M_P, verbose=False, num_steps=1000)
solver1 = rt.RadialSolver(V_HO, (rt.H_BAR*omega-V1+5*MEV,-20*MEV), 1, 0.025*R, M_P, verbose=False, num_steps=1000)
solver2 = rt.RadialSolver(V_HO, (rt.H_BAR*omega-V1+10*MEV,-20*MEV), 2, 0.025*R, M_P, verbose=False, num_steps=1000)
solver3 = rt.RadialSolver(V_HO, (rt.H_BAR*omega-V1+17*MEV,-20*MEV), 3, 0.025*R, M_P, verbose=False, num_steps=1000)
plt.plot(solver0.rgrid/R, V_HO(solver0.rgrid)/MEV, marker='.')
plt.xlabel('r/a')
plt.ylabel('$V(r)$')
plt.show()

# # Plot kink
print("Plotting kink vs. E...")
Epoints=np.linspace(rt.H_BAR*omega-V1,-2*MEV,500)
k0points=[]
k1points=[]
k2points=[]
k3points=[]
for ep in Epoints:
    try:
        k0points.append(solver0.calculate_kink(ep))
    except:
        k0points.append(0)
    try:
        k1points.append(solver1.calculate_kink(ep))
    except:
        k1points.append(0)
    try:
        k2points.append(solver2.calculate_kink(ep))
    except:
        k2points.append(0)
    try:
        k3points.append(solver3.calculate_kink(ep))
    except:
        k3points.append(0)

plt.xlabel('E')
plt.ylabel('$\psi^\prime_{right}(c) - \psi^\prime_{left}(c)$')
plt.plot(Epoints/MEV,k0points, marker='.', label="l = 0")
plt.plot(Epoints/MEV,k1points, marker='.', label="l = 1")
plt.plot(Epoints/MEV,k2points, marker='.', label="l = 2")
plt.plot(Epoints/MEV,k3points, marker='.', label="l = 3")
plt.show()

# Actually solve and plot
energies = []
solver = rt.RadialSolver(V_HO, (-40*MEV,-35*MEV), 0, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0s')
energies.append(energy)
print("E(0s)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-35*MEV,-25*MEV), 1, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0p')
energies.append(energy)
print("E(0p)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-30*MEV,-20*MEV), 2, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0d')
energies.append(energy)
print("E(0d)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-25*MEV,-15*MEV), 3, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='0f')
energies.append(energy)
print("E(0f)=",energy/MEV)

solver = rt.RadialSolver(V_HO, (-30*MEV,-20*MEV), 0, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='1s')
energies.append(energy)
print("E(1s)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-25*MEV,-15*MEV), 1, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='1p')
energies.append(energy)
print("E(1p)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-15*MEV,-5*MEV), 2, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='1d')
energies.append(energy)
print("E(1d)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-10*MEV,-1*MEV), 3, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='1f')
energies.append(energy)
print("E(1f)=",energy/MEV)

solver = rt.RadialSolver(V_HO, (-15*MEV,-5*MEV), 0, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='2s')
energies.append(energy)
print("E(2s)=",energy/MEV)
solver = rt.RadialSolver(V_HO, (-10*MEV,-1*MEV), 1, 0.025*R, M_P, verbose=False, num_steps=2000)
energy = solver.solve(tolerance=1e-12)
plt.plot(solver.rgrid*1e15, solver.solution_points, marker='.', label='2p')
energies.append(energy)
print("E(2p)=",energy/MEV)


for i in range(len(energies)-1):
    print((energies[i+1]-energies[i])/MEV)

plt.legend()
plt.title('Woods-Saxon s-wave functions')
plt.xlabel('r/a')
plt.ylabel('$\psi(r)$')
plt.show()
