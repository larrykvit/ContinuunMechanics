import numpy as np
from numpy import einsum as es
from numpy import tensordot as td

import scipy.linalg as spla

#the helper functions needed for the assignment
DIMS = 3 #cause 3D stuff, probably will break if you change this

#predefined identities
kdel = np.identity(DIMS) #kronecker delta

eye4 = es('ik,jl->ijkl',kdel,kdel)
eye4s = 0.5*(es('ik,jl',kdel,kdel) + es('il,jk',kdel,kdel))
eye4vol = es('ij,kl',kdel,kdel)/3
eye4dev = eye4s - eye4vol

def perpendicular(a, b):
    '''find normalized pernendicualr vector'''
    return np.linalg.norm(np.cross(a,b))


# predefined multiplications
def tp_1dy1(a,b):
    '''tensor product between two scalars, result is dyad'''
    return np.tensordot(a,b, axes=0);

def tc_1s1(a, b):
    '''tensor contraction between two 1d vectors'''
    return np.tensordot(a,b, axes = 1)
    #return np.einsum('i,i', a, b)

def tc_2d2(a, b):
    '''tensor contraction between 2 matrix'''
    return np.tensordot(a,b)

def tc_4d2(a, b):
    '''tensor contraction between 4th 2nd'''
    return np.tensordot(a,b)

def tc_4d4(a, b):
    '''tensor contraction between 4th 4th'''
    return np.tensordot(a,b)

def stress_eq_vm(sigma):
    '''find equivalent von mises equilvalent stress'''
    s_dev = stress_dev(sigma)
    return np.sqrt(1.5 * np.tensordot(s_dev, s_dev))


def stress_dev(sigma):
    ''' return the devitoric stress'''
    return td(eye4dev, sigma)


def stress_hyd(sigma):
    '''return hydrostatic stress'''
    return td(eye4vol, sigma)


def elestic_stress(E, v):
    '''4th order elastic tensor'''
    #shear modulus
    G = E/(2*(1+v));
    #bulk modulus
    K = E/(3*(1-2*v));
    #lame constant
    l = K-2*G/3;

    return l*eye4 + 2*G*eye4s;

def elestic_compliance(E, v):
    '''4th order elastic compliance tensor'''
    #shear modulus
    G = E/(2*(1+v));
    #bulk modulus
    K = E/(3*(1-2*v));

    return (-v/E) * eye4 + (1/(2*G)) * eye4s


#the different invariants and such
#deviatoric multiply A by the deviatoric tensor to get s, then do the same
#J1(A) = 0 (deviatoric stress has trace == 0)

def I1(A):
    return np.trace(A)

def I2(A):
    return (np.trace(A)**2 - np.trace(A@A))/2

def I3(A):
    return np.linalg.det(A)

def J1(A): return 0
def J2(A): return -1*I2(stress_dev(A))
def J3(A): return I3(stress_dev(A))

def triaxiality(s):
    '''the triaxiality is for the severity of the volumetric loading'''
    return (I1(s)/3)/np.sqrt(3*J2(s))


def lode_parameter(s):
    '''The Lode parameter is used to indicate the severity of the deviatoric (shear) loading'''
    return (-27/2)*J3(s)/(stress_eq_vm(s)**3)


def cayley_hamilton(A):
    '''calculate eigen values using cayley-hamilton'''
    H1 = I1(A)/3
    H2 = -I2(A)/3
    H3 = I3(A)/2

    p = H1**2 + H2
    q = (2*H1**3 + 3*H1*H2 + 2*H3)/2
    th = np.arccos(q/p**1.5)

    s = [2*np.sqrt(p)*np.cos((th+ o*np.pi)/3 ) + H1 for o in range(0, 5, 2)]
    return s


def normal_von_mises(s):
    '''normal tensor for the fon mises yield function'''
    s_dev = stress_dev(s)
    s_vm = stress_eq_vm(s)
    N = 1.5 * s_dev / s_vm
    return N


def normal_logan_hosforf(s1, s2, a, R0, R90):
    A = R90/(1+R0)
    B = R0/(1+R0)
    C = R0*R90/(1+R0)

    d = (s1-s2)

    N1 = (A*s1**a+B*s2**a+C*d**a)**(1/a-1)*(a*A*s1**(a-1) + a*C*d**(a-1))/a
    N2 = (A*s1**a+B*s2**a+C*d**a)**(1/a-1)*(a*B*s2**(a-1) + a*C*d**(a-1))/a

    N3 = -(N1+N2)

    return (N1, N2, N3)


def green_largrange_stretch_strain(F):
    '''green-lagrange stretch and strain C, E'''
    C = F.transpose()@F
    E = 0.5*(C - np.identity(DIMS))

    return (C, E)


def velocity_gradient(F, Fdot):
    '''velocity gradient and deformation tensor and vortiticy L D W'''
    L = Fdot @ np.linalg.inv(F)
    D = 0.5*(L+L.transpose())
    W = 0.5*(L-L.transpose())

    return (L, D, W)


def polar_decomposition(F):
    '''decompose F into rotation and stretch of both kinds R, V, U'''
    (R, U) = spla.polar(F)
    (_, V) = spla.polar(F, 'left')

    return (R, V, U)

def objective_update(A, Spin, Adotobj):
    '''objective update of A with a certain spin, and objective rate of A'''
    Adot = Adot - A @ Spin + Spin @ A
    Anext = A + Adot

    return Anext

UNIAXIAL_TENSION = 'uniaxial tension'

#define some test case stress tensors
sample_stress = {
    'uniaxial compression'  :np.diag((-1,   0, 0)),
    'uniaxial tension'      :np.diag(( 1,   0, 0)),
    'equal biaxial'         :np.diag(( 1,   1, 0)),
    'plane strain'          :np.diag(( 1, 0.5, 0)),
    'simple shear'          :np.fliplr(np.diag([1,1],k=1)),
}

