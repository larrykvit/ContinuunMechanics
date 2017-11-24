from continuumMechanics import *
import matplotlib as mpl
import matplotlib.pyplot as plot

#question 2
print('question 2')
F = np.diag((1,1,1))
F[0,1] = 6

Fdot = np.diag((0.5,0),k=1)


#part a
print('part a')
(C, E) = green_largrange_stretch_strain(F)
print(C)
print(E)

#part b
print('part b')
(L, D, W) = velocity_gradient(F, Fdot)
print(L)
print(D)
print(W)

#part c
print('part c')
(R, V, U) = polar_decomposition(F)
print(R)
print(V)
print(U)


#question 5
print('question 5')
gamma = 12
dt = 1/500
dgamma = gamma *dt #assume it happens in 1 second

gammas = np.linspace(dgamma, gamma, 1/dt)

plotStrains = []

strain = np.zeros((DIMS, DIMS))


for cur_gamma in gammas:
    F = deformation_gradient_simple_shear(cur_gamma)
    Fdot = deformation_gradient_dot_simple_shear(dgamma/dt)

    (L, D, W) = velocity_gradient(F, Fdot)

    strain_dot_obj = D

    #jaumann spin rate
    #spin = W

    spin = log_spin(dgamma/dt, cur_gamma)

    strain = objective_update(strain, spin, strain_dot_obj, dt)

    e11 = strain[0,0]
    e22 = strain[1,1]
    e1 = max(np.linalg.eigvals(strain))

    plotStrains.append([e11, e22, e1])

plotStrains = np.array(plotStrains)

fig_log = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(gammas, plotStrains)
ax.plot(gammas, np.arcsinh(gammas/2), 'r--')
ax.legend([r'$\epsilon_{11}$', r'$\epsilon_{22}$', r'$\epsilon_{1}$', r'Closed-form $\epsilon_{1}$'])

ax.set_title('Logarithmic Spin')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\epsilon$')

plot.show(fig)
print('logarithmic strain')
print(strain)
print('principle direction')
d = 0.5*np.arctan((strain[0,0]-strain[1,1])/(strain[0,1]))
print(d)


#jaumann spin
plotStrains = []
strain = np.zeros((DIMS, DIMS))
for cur_gamma in gammas:
    F = deformation_gradient_simple_shear(cur_gamma)
    Fdot = deformation_gradient_dot_simple_shear(dgamma/dt)

    (L, D, W) = velocity_gradient(F, Fdot)

    strain_dot_obj = D

    #jaumann spin rate
    spin = W

    strain = objective_update(strain, spin, strain_dot_obj, dt)

    e11 = strain[0,0]
    e22 = strain[1,1]
    e1 = max(np.linalg.eigvals(strain))

    plotStrains.append([e11, e22, e1])


plotStrains = np.array(plotStrains)

fig_j = plot.figure()
ax = fig_j.add_subplot(1,1,1)
ax.plot(gammas, plotStrains)
ax.plot(gammas, np.arcsinh(gammas/2), 'r--')
ax.legend([r'$\epsilon_{11}$', r'$\epsilon_{22}$', r'$\epsilon_{1}$', r'Closed-form $\epsilon_{1}$'])

ax.set_title('Jaumann Spin')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\epsilon$')

plot.show(fig_j)

print('jaumann strain')
print(strain)
print('principle direction')
d = 0.5*np.arctan((strain[0,1])/(strain[0,0]-strain[1,1]))
print(d)
print('done')