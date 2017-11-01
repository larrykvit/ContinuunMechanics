from continuumMechanics import *

# question 2
print('Question 2')

fmt = '{:<26}:{:>5}    {:>5}'
print(fmt.format('', 'T', 'L'))
for name, stress in sample_stress.items():
    T = triaxiality(stress)
    L = lode_parameter(stress)
    print(fmt.format(name, round(T, 4), round(L, 4)))


#question 3
print()
print()
print('Question 3')
stress = np.array( [[100, 10, -10],
                    [10 , -5,  20],
                    [-10, 20,  50]])

print('Solving with Cayley-Hamilton')
s = cayley_hamilton(stress)
print(s)

print('Checking against numpy eigvals')
e = np.linalg.eigvals(stress)
print(e)

print(np.allclose(s,e))


#question 4
print()
print()
print('Question 4')
t = np.random.rand(DIMS, DIMS)
stress = t + t.transpose()

print('testing von mises normal vector with random matrix')
print(stress)

N = normal_von_mises(stress)
print('result of N:N = {}'.format(td(N, N)))


print('testing with uniaxial tensile test')

stress = sample_stress[UNIAXIAL_TENSION]
N = normal_von_mises(stress)
print('R-value N22/N33 = {}'.format(N[1,1]/N[2,2]))
print('Testing incompressibility: N11 + N22 + N33 = {}'.format(np.trace(N)))


