import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def f(y,t,mstar):
    G = 6.67408e-08 # Gravitational constant in CGS units [cm^3/g/s^2]
    gm = G*mstar
    x = y[0:3].copy()
    v = y[3:6].copy()
    r = np.sqrt(x[0]**2+x[1]**2+x[2]**2)
    dxdt = v
    dvdt = -gm*x/r**3
    dy = np.hstack((dxdt,dvdt))
    return dy
msun = 1.98892e33 # Solar mass [g]
year = 31557600.e0 # Year [s]
au = 1.49598e13 # Astronomical Unit [cm]
t0 = 0. # Starting time
tend = 1.5*year # End time
nt = 100 # Nr of time steps
t = np.linspace(t0,tend,nt)
r0 = au # Initial distance of Earth to Sun
vp0 = 0.8 * 2*np.pi*au/year # Initial velocity of Earth
x0 = np.array([r0,0.,0.]) # 3-D initial position of Earth
v0 = np.array([0.,vp0,0.]) # 3-D initial velocity of Earth
y0 = np.hstack((x0,v0)) # Make a 6-D vector of x0 and v0
sol = odeint(f,y0,t,args=(msun,))
x = sol[:,0:3].T
v = sol[:,3:6].T
plt.figure()
plt.plot(x[0,:]/au,x[1,:]/au)
plt.plot([0.],[0.],'o')
plt.xlabel('x [au]')
plt.ylabel('y [au]')
plt.axis('equal')
9
plt.savefig('fig_kepler_1_1.pdf')
plt.show()
