import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plot


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

# the grid and center points: 0 1/2N 1/N ... .... ... (N-1)/N (N-1)/N + 1/(2N) 1 1 + 1/(2N)
n = 50 # to start
h = 1.0/n # space grid resolution
x = np.arange(h/2,1+h/2,h)
f = Q*(1+S2*P2(x))*(a0 + a2*P2(x)) - A # a forcing function evaluated on grid points.

Tmax = 30 # in years 
dtinv = 12 # time grid resolution (inverse)
t  = np.linspace(0,Tmax,dtinv*Tmax)

## task 1: simple solution

T0 = 10*np.ones(x.shape) # initial temp. profile.

sol = odeint(forced_heat, T0, t, args=(h,B,D,cw)) 

# plot results
plot.plot(t,sol)
plot.xlabel('t, years')
plot.ylabel('temp, celcius')
plot.title('temperature evolution')
plot.show()

plot.plot(x,sol[Tmax*dtinv-1,])	# temp profile at final time
plot.xlabel('x, degrees lat')
plot.ylabel('temp, celcius')
plot.title('steady-state temperature profile')
plot.show()

# results: plots that give qualitatively correct temperature distribution, but with 
# greater variance on the high end, toward 60 celcius. 

# rhs of a forced heat equation with diffusion given below.
# Note: f, T are (N+1)x1 vectors below. you need all (or, at least the
# nearest neighbors) of T in order to define operator diffusion(T). 
def forced_heat(T, t, h, B, D, cw):
	return 1/cw*( f - B*T + diffusion(T,h,D))

# test with: plot.plot(x,forced_heat(x,1,h,2, 2,1))

# The spatially inhomogenous diffusion operator
def diffusion(u,h,D):
	# u is a N+1x1 vector: 0, 1, ..., N-1, N.
	N = u.shape[0]-1
	
	diffu = np.zeros(N+1) # initialize as (N+1)x1 vector
	
	# at i=1, use boundary condition to rewrite diffusion operator
	# as a leapfrog diference. Need N \geq 2.
	diffu[1] = 2*D*(u[2] - u[1])/h   
	# central difference/leapfrog for interior point
	for i in np.arange(2, N):
		diffu[i] = D*( (1-x[i]**2)*cdiff2(u,i,h) - 2*x[i]*leap(u,i,h) )
	
	# Use boundary condition to write endpoint at pole. Note higher-order
	# scheme will increase accuracy. Q: Is this the O(h) bottleneck without
	# which we could upgrade to O(h^2) globally? Does it matter that much
	# practically since on average global error is still O(h^2) given 
	# central difference scheme usd in bulk? Maybe this just reduces to 
	# "errors depends on choice of norm".
	diffu[N] = -2*(u[N] - u[N-1])/h
	
	return diffu
    

def leap(u,i,h):
    return (u[i+1] - u[i-1])/(2*h)

def cdiff2(u,i,h):
    return (u[i+1] - 2*u[i] + u[i-1])/(h**2) 

# testing: cdiff2(x,i,h) ~ 0, i.e. second deriv of line = 0
#		   leap(x,i,h) ~ 1, i.e. first deriv of line = 1

## Task 2: implicit time-stepping and staggered grid

# staggered grid:
h = 1.0/n # grid box width
x = np.arange(h/2,1+h/2,h) #native grid
xb = np.arange(h,1,h)

# build diffusion matrix
lam = D/dx**2*(1-xb**2)
L1=np.append(0, -lam)
L2=np.append(-lam, 0)
L3=-L1-L2
diffop = - np.diag(L3) - np.diag(L2[:n-1],1) - np.diag(L1[1:n],-1)




