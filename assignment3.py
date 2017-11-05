from continuumMechanics import *


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
