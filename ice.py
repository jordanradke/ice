import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


## Initialize parameters

Li = 9.5 # sea ice latent heat of fusion [W/(m^3 years)]
coHo = 6.3 # heat capacity of OML [W/(m^2 years K)]
zeta = 0.7 # sea ice thermodynamic scale thickness [m]

# Atmosphere/ice interface
a_bar   = .56  # co-albedo, ice/ocean average
delta_a = .48  # difference of ice/ocean co-albedos
h_a     = .5   # smoothness of albedo transition, [m]

S_m = 100  # short-wave (solar) flux: annual mean [W/m^2]
S_a = 150  # short-wave flux: amplitude of seasonal variation [W/m^2]

L_m = 70   # long-wave flux: annual mean [W/m^2]
L_a = 41   # long-wave flux: seasonal amplitude [W/m^2]
phi = 0.15 # phase shift: summer solstice - peak longwave forcing [years]
p   = 1  # period of seasonal forcing term [years]

B = 2.83 # net surface flux per surface degree [W/(m^2 K)]

# Ice/ocean interface
F_b = 0 # heat flux at bottom


## Integrate ODE for enthalpy

# Returns the time-dependent forcing term A.
def A(E, t, a_bar, delta_a, h_a, Li, S_m, S_a, L_m, L_a, p, phi, X):
	return ((a_bar + delta_a/2*np.tanh(E/(Li*h_a)))*(S_m -S_a*np.cos(2*np.pi*t/p)) - L_m - X - 	
		L_a*np.cos(2*np.pi*(t-phi)/p))


# To recover surface temperature from enthalpy.
def surface_temp(E, coHo, A, B, Li, zeta):
    if E >= 0:
        return E/coHo
    elif E < 0 and A >= 0:
        return 0
    else:
        return A/B*((1-zeta*Li/E)**(-1))
        
        
# computes the right-hand side of the ODE for enthalpy. Sign error fixed 10/16
def rhs(E, t, a_bar, delta_a, h_a, Li, S_m, S_a, L_m, L_a, p, phi, coHo, B, zeta, F_b, X):
    a = A(E, t, a_bar, delta_a, h_a, Li, S_m, S_a, L_m, L_a, p, phi, X)
    return (a - B*surface_temp(E, coHo, a, B, Li, zeta) + F_b)


## TASK 1: Evolve enthalpy over 50 year time span; compare warm vs. cold initial condition.

# Note steady states have seasonal variation.

# Now, evolve enthalpy according to $\frac{dE}{dt} = f(t,E)$ where the forcing is given by 
# $f(t,E)  = A - B T(E) + F_b$. The departure from surface temperature, $T$, is given in terms of 
# the enthalpy above.
def enthalpy_int(E_0, h_a, t_max, res, X=0):
	# parameters:
	# E_0: initial enthalpy
	# h_a: hysteresis parameter
	# t_max: time period [years]
	# res: time grid resolution [years]
	t = np.linspace(0,t_max,res) # integrate over 50 years, with monthly grid spacing	
	sol = (odeint(rhs, E_0, t, args=(a_bar, delta_a, h_a, Li, S_m, S_a, L_m, L_a, p, phi, coHo, B, zeta, 
		F_b, X)))
	return sol
    

res = 12*100 # time resolution
t_max = 50	 # max time (years)
t = np.linspace(0,t_max,res)

E_0w = 100 # initial condition for enthalpy [J/m^2]
E_0c = -100 # the cold case.

solw = enthalpy_int(E_0w, h_a, t_max, res)
solc = enthalpy_int(E_0c, h_a, t_max, res)

## Plot results


plt.plot(t, solw[:, 0], 'b', label='warm')
plt.plot(t, solc[:, 0], 'r', label='cold')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.title('enthalpy evolutions')
plt.show()


## TASK 2: Represent long-wave radiative (negative) forcing in above model 

X=10
solw = enthalpy_int(E_0w, h_a, t_max, res, X)
solc = enthalpy_int(E_0c, h_a, t_max, res, X)

plt.plot(t, solw[:, 0], 'b', label='warm')
plt.plot(t, solc[:, 0], 'r', label='cold')
plt.legend(loc='best')
plt.xlabel('t')
plt.grid()
plt.title('enthalpy evolutions with forcing')
plt.show()











