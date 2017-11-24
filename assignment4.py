from continuumMechanics import *
import matplotlib as mpl
import matplotlib.pyplot as plot


#question 1
print('Question 1')
E = 205 * 10**9
v = 0.25

spins = ['log', 'jaumann']
principle = {}

for spin in spins:
    gammas, strains, stresses, angles = shear_integration(
        4, 1/500, obj_rate=spin, E=E, poisson=v)

    anal_stresses = np.array([shear_analytical(g, E, v) for g in gammas])

    G = E/(2*(1+v))

    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(gammas, stresses/G)
    ax.plot(gammas, anal_stresses/G, '--')
    ax.legend([r'$\sigma_{11}$', r'$\sigma_{12}$', r'$\sigma_{1}$',
               r'$analytical \sigma_{11}$', r'$analytical \sigma_{12}$', r'$analytical \sigma_{1}$'])

    ax.set_title(spin + ' spin')
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$\sigma/G$')

    plot.show(fig)

    principle[spin] = angles

fig = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(gammas, principle['log'], label='Log spin')
ax.plot(gammas, principle['jaumann'], label='Jaumann spin')
ax.legend()
ax.set_title('Comparing principle direction with different integration')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel(r'$\theta$')
plot.show(fig)

print('---')
print('')

#question 4
print('Question 4')

#hyper
gammas, strains_hyper, stresses_hyper = shear_integration_hyper(4, 1/500, E=E, poisson=v)

gammas, strains_log, stresses_log, angles = shear_integration(4, 1/500, obj_rate='log', E=E, poisson=v)
gammas, strains_jau, stresses_jau, angles = shear_integration(4, 1/500, obj_rate='jaumann', E=E, poisson=v)

#part i
fig = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(gammas, strains_hyper[:,0:2])
ax.plot(gammas, strains_log[:,0:2])
ax.plot(gammas, strains_jau[:,0:2])
ax.legend([ r'$E_{11}$', r'$E_{12}$',
            r'$\epsilon_{11}$ log', r'$\epsilon_{12}$ log',
            r'$\epsilon_{11}$ jaumann', r'$\epsilon_{12}$ jaumann'])
ax.set_title('Strain vs gamma')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('Strain')
plot.show(fig)

#part ii
fig = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(gammas, stresses_hyper[:,0:2]/G)
ax.plot(gammas, stresses_log[:,0:2]/G)
ax.plot(gammas, stresses_jau[:,0:2]/G)
ax.legend([ r'$S_{11}$', r'$S_{12}$',
            r'$\sigma_{11}$ log', r'$\sigma_{12}$ log',
            r'$\sigma_{11}$ jaumann', r'$\sigma_{12}$ jaumann'])
ax.set_title('Stress vs gamma')
ax.set_xlabel(r'$\gamma$')
ax.set_ylabel('Stress / G')
plot.show(fig)

#part iii
fig = plot.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(strains_hyper[:,0:2], stresses_hyper[:,0:2]/G)
ax.plot(strains_log[:,0:2], stresses_log[:,0:2]/G)

ax.legend([ r'$S_{11}$', r'$S_{12}$',
            r'$\sigma_{11}$ log', r'$\sigma_{12}$ log'])
ax.set_title('Stress vs strain')
ax.set_xlabel('Strain')
ax.set_ylabel('Stress / G')
plot.show(fig)

print('---')
print('')
