import numpy as np
import matplotlib.pyplot as plot

### Task 2: implicit time-stepping and staggered grid
#
# Main difficulties:
# Q1. How to implement spatially non-homogeneous diffusion  
#       operator.
# A1. use staggered grid
# Q2. Dealing with boundary conditions, one of which is null.
# A2. backwards diff which agrees with our diffusion operator
#       at polar end point, extra grid point enforces no-flux 
#       condition at equator end point.
#
# Evolve the forced heat equation
# \D_t u = f(x) - Bu + D \Div( (1-x^2) \nabla_x u )     (forced_heat)
# subject to
# BC1: at x=0: \D_x u(t,0) = 0          (noflux_eq)
# BC2: at x=1: (1-x^2)\D_x u(t,1) = 0   (noflux_pole).
# 
#   Q: Finite element methods + divergence form structure. How to
#       enforce conservation laws numerically.
#
# First, discretize in space (method of lines): Let U_i(t) denote
# the ith grid-point sample of U at time t, for i=0,...,N+1, 
# each subject to the initial condition U_i(0) = U^0_i. Given 
# some 'continuous' data (or, data defined on a grid point mesh
# with h' << h, the current mesh) we set U^0_i to be the average
# value of the field U^0(x) in this grid cell of diameter h 
# centered at x(i). Then, for each i, we wish to solve
#
# \D_t U_i(t) = f(x) - BU_i(t)
#             + D\{ (1-x_b(i)) h**(-2) (U_{i+1} - 2U_i + U_{i-1})
#                   -2*x_b(i)* (2h)**(-1) (U_{i+1} - U_{i-1})     
#   subject to  U^0_i(0) = U^0_i,                             (*)
# 
# where we have used a centered difference scheme for the second-
# order derivative in the diffusion term, and a leapfrog scheme
# for the first-order spatial derivative. We have defined
#
# f(x) = Q*(1+S2*P2(x))*(a0 + a2*P2(x)) - A,
#   Q: Utility of evaluating forcing at function staggered grid?
#
# with parameters defined as in [NorthCahalanCoakley81].
# Here, as before, the time-independent forcing term is determined
# by the incoming and outgoing radiative forcing, as well as the -B
# term, which should represent exponential decay in time, all other
# things equal (i.e. if the scales of the forcing and diffusion
# terms were small enough so that to a first approximation the 
# above equation (*) would be \D_t U_i(t) = -B U_i(t), the decoupled 
# ODE system with solutions exp(-Bt)*U^0_i.)
#
# Now we introduce the time differencing scheme (implicit euler):
# Let k = 1 ,..., dtinverse*Tmax, with t(k) the kth time step of
# our iteration. Discretize \D_t U as 
#
# U_i(k+1) - U_i(k) = c_w**(-1)*dt*rhs( U(k+1) )        (implicit_euler)
#
# where rhs() denotes the r.h.s of (*), evaulated at time step k+1. 
#
#
# The individual terms comprising rhs( . ): the first two
# are constant force of Nx1 matrix f, the second scalar multiplication of 
# the solution by B, represented by np.diag(B*np.ones(N)). For the diffusion
# term, in which we enforce (BC1) , we define 
# a matrix operator below, diffusion[U] which takes in the (N+2)x1 vector
# U and outputs a (N+2)x1 vector; the boundary condition BC1 is 
# enforced at the equator grid point by a one-sided diff scheme at 
# U1. The boundary condition BC2 presents no information on the
# derivative \D_x u(t,1), so we update the last grid point
# according to a one-sided difference scheme
#
# (3 -4 1)/(2h) 0 0 0 0 0 0      (BC1: enforce noflux_eq)
#
# np.dot(coeff1,toep) - np.dot(coeff2,leap)
#
# 0 0 0 0 0 0 0 (1 -4 3)/(2*h)   (one_sidedDiffusion-last int pt)
# 0 0 0 0 0 0 0 0 0 0 0 0 0      (BC2: ?)
#
# Note that the second-order-in-space term dissappears at this
# boundary point. So the last row of the second-order term (1-x^2)d^2u/dx^2
# in diffusion[U] will evaluate to zero no matter the ambiguity at
# the boundary. Upshot: just insert lastrow = [0 0 ... 0 0 ] as
# argument to function Toep( . ) and come to this point later.

## INITIALIZE PARAMETERS
Q = 340;         # solar constant/4 = 1360/4 (W m^-2)
A = 203.3;       # Outgoing Long-wave Radiation (OLR) when T = 0 (W m^-2)
B = 2.09;        # OLR temperature dependence (W m^-2 K^-1)
a0 = 0.681;      # 1. Legendre Poly. albedo cfft. An order 2 expansion.
a2 = -0.202;     # 2. Legendre Poly. albedo cfft
S2 = -0.477;     # solar forcing value in NCC81 (W m^-2)
cw = 6.3;        # ocean mixed layer heat capacity (W yr m^-2 K^-1)
D = 0.649;       # diffusivity for heat transport (W m^-2 K^-1)

# co-alebedo and insolation are approximated with Legendre polynomials.
P2 = lambda X: 1/2*(3*X**2-1); # define as function lambda.

## SPACE AND TIME GRID
# space grid and center points: 0 1/2N 1/N ... .... ... (N-1)/N (N-1)/N + 1/(2N) 1 1 + 1/(2N)
n = 10 # to start
h = 1.0/n # space grid resolution
x = np.concatenate( (np.zeros(1), np.arange(h/2,1+h/2,h), np.ones(1)) )
f = Q*(1+S2*P2(x))*(a0 + a2*P2(x)) - A # a forcing function evaluated on grid points.
# boundaries don't see forcing?
#f[0] = 0
#f[n+1] = 0
# We define the staggered grid to sample diffusion coefficient at 
# higher accuracy near boundaries 
xb = np.linspace(0,1,n+2)

# time
Tmax    = 30 # in years 
dtinv   = 100 # time grid resolution (inverse)
t       = np.linspace(0,Tmax,dtinv*Tmax)

## build diffusion matrix
#l = (1-xb**2)   # sample coefficients from the staggered grid. 
# Q: should be (1-xb**2)^(1/2)?

# BCs to cdiff2: for second-order derivatives // enforcing (BC1)
row0    = np.append( .5*np.asarray([3,-4,1]), np.zeros(n+2-3) )
rown1   = np.zeros(n+2) # [bc2_null]

# BCs to cdiff1: 
rowa0    = np.zeros(n+2)
# backwards difference for boundary point
rowan1   = np.append( np.zeros(n+2-3), .5*np.asarray([1,-4,3]) )


# Given representation of boundary behavior, create finite diff operators
# at grid resolution h. Note BCs come -without- scale. Nuemann enforced at
# scale h, Dirichlet enforced at scale 1. TODO: rewrite these functions
# (see numpy, scipy for analogues)
    
cdiff2 = Cdiff2(row0,rown1,h)
cdiff1 = Cdiff1(rowa0,rowan1,h)

diffusion = D*(np.dot(coef2(xb), cdiff2) + np.dot(coef1(xb), cdiff1))

## Solve system
# set up inverse problem by solving (implicit_euler) for U(k+1):
# U(k+1) = U(k) + c_w**(-1)*dt*f - c_w**(-1)*dt*(B*I - diffusion)U(k+1) 
# => (I - cw**(-1)*dt*(B*I - diffusion))U(k+1) = U(k) + c_w**(-1)*dt*f
# then define M = ( l.h.s matrix ) and invert to update at U(k+1) step.
U = np.zeros((dtinv*Tmax,n+2))    # initialize solution matrix
Uavg = np.zeros(dtinv*Tmax)


U[0,] = 10*np.ones(n+2) # initial temp profile
Uavg[0] = np.average(U[0,])

dt = dtinv**(-1)
I_nminus2 = np.eye(n+2)
I_nminus2[0,0]=0
# I_nminus2[n+1,n+1]=0


M = I_nminus2 + cw**(-1)*dt*(B*I_nminus2 - diffusion)

for k in np.arange(dtinv*Tmax-1):
	U[k+1,] = np.dot(np.linalg.inv(M), U[k,] + cw**(-1)*dt*np.dot(I_nminus2,f))
	Uavg[k+1] = np.average(U[k+1,])
	if k % 10 == 0:
		print(Uavg[k+1])
	


plot.plot(x,U[dtinv*Tmax-1,])
plot.show()


### Plots and testing against known analytic solution.

# diffusion coefficient as function of latitude. Note higher 
# diffusivity at equator, simulating mixing effects of tropical
# convection? 
#plot.plot(xb,l)
#plot.title("Diffusion coefficient as function of latitude")
#plot.xlabel("sin(latitude)")
#plot.ylabel("diffusion coefficient")
#plot.show()



## HELPER FUNCTIONS FOR ABOVE
# to define diffusion finite diff matrix
def Cdiff2(row0,rown1,h):
	# Inputs: the 0, 1, N, N+1 -th rows of a matrix.
	#   row0: enforces boundary condition, if any.
	#   rown: enforces boundary condition, if any.
	#   h: the grid resolution of finite diff scheme.
	n = row0.shape[0] - 2
	cdiff2 = np.zeros( (n+2,n+2) )
	
	cdiff2[0,] = h*row0  # h since nuemann
	for i in np.arange(n):
		cdiff2[i+1,] = h**(-2)*np.concatenate( (np.zeros(i), np.asarray([1,-2,1]), np.zeros(n-i-1)) )
	
	cdiff2[n+1,] = h*rown1 # 
	return cdiff2


def Cdiff1(row0, rown1,h):
	# Inputs: the 0, 1, N, N+1 -th rows of a matrix.
	#   row0: enforces boundary condition, if any.
	#   rown: enforces boundary condition, if any.
	#   h: the grid resolution of finite diff scheme.
	n = row0.shape[0] - 2
	cdiff1 = np.zeros( (n+2,n+2) )
	
	cdiff1[0,] = h*row0
	for i in np.arange(n):
		cdiff1[i+1,] = .5*h**(-1)*np.concatenate( (np.zeros(i), np.asarray([1,0,-1]), np.zeros(n-i-1)) )
	
	cdiff1[n+1,] = h*rown1
	return cdiff1


# M is an NxN matrix
#def Buff(M,a=1):
#    
#    N = M.shape[0]
#    M1 = np.zeros( (N+2,N+2) )
#    
#    M1[0,] = np.concatenate( (np.asarray([1]), np.zeros(N+1)) )
#    for i in np.arange(N):
#        M1[i+1,1:N+1] = M[i,]
#        
#    M1[N+1,] = np.concatenate( (np.zeros(N+1), np.asarray([1])) )
#    return M1


# second-order deriv coefficient matrix    
def coef2(xb):
	return np.diag(1-xb**2) # note zero at xb(N+1) = 1


# first-order deriv coefficient matrix
def coef1(xb):
	return np.diag(-2*xb)



