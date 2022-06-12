import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
def f(y,t,m):
    G = 6.67408e-08 # Gravitational constant in CGS units [cm^3/g/s^2]
    n = len(m)
    plan = y.reshape((n,6))
    frc = np.zeros((n,3))
    dy = np.zeros((n,6))
    
    for i in range(n):
        x = plan[i,0:3]
        for j in range(i+1,n):
            x1 = plan[j,0:3]
            dx = x1-x
            r = np.sqrt(dx[0]**2+dx[1]**2+dx[2]**2)
            df = G*m[i]*m[j]*dx/r**3
            frc[i,0:3] += df[0:3]
            frc[j,0:3] -= df[0:3]
    for i in range(n):
        x = plan[i,0:3].copy()
        v = plan[i,3:6].copy()
        dxdt = v
        dvdt = frc[i,0:3]/m[i]
        dy[i,0:6] = np.hstack((dxdt,dvdt))
    return dy.reshape((6*n))
G = 6.67408e-08 # Gravitational constant in CGS units [cm^3/g/s^2]
Msun = 1.98892e33 # Solar mass [g]
Mju = 1.899e30 # Mass of Jupiter [g]
year = 31557600.e0 # Year [s]
au = 1.49598e13 # Astronomical Unit [cm]
t0 = 0. # Starting time
tend = 4.5*year # End time
nt = 400 # Nr of time steps
t = np.linspace(t0,tend,nt)
m = np.array([Msun,Mju,Mju]) # Masses
a0 = np.array([0.,au,1.15*au,]) # Semi-major axes
phi0 = np.array([0.,0.,0.]) # Orbital locations in degrees
nb = len(m)
x0 = np.zeros((nb,3))
v0 = np.zeros((nb,3))
pos = np.zeros(3)
mom = np.zeros(3)
for i in range(1,nb): # Loop over all planets (i.e. excluding star)
    x0[i,0] = a0[i]*np.cos(phi0[i])
    x0[i,1] = a0[i]*np.sin(phi0[i])
    vphi = np.sqrt(G*m[0]/a0[i])
    v0[i,0] = -vphi*np.sin(phi0[i])
    v0[i,1] = vphi*np.cos(phi0[i])
    pos[:] += m[i]*x0[i,:]
    mom[:] += m[i]*v0[i,:]
x0[0,:] = -pos[:]/m[0] # Total center of mass must be at (0,0,0)
v0[0,:] = -mom[:]/m[0] # Total momentum must be (0,0,0)
xv0 = np.zeros((nb,6))
for i in range(nb):
    xv0[i,0:3] = x0[i,0:3]
    xv0[i,3:6] = v0[i,0:3]
y0 = xv0.reshape(6*nb) # The full nb*6 element vector
sol = odeint(f,y0,t,args=(m,)) # Solve the N-body problem
xv = sol.reshape((nt,nb,6)) # Now extract again the x and v
x = np.zeros((nb,3,nt))
v = np.zeros((nb,3,nt))
for i in range(nb):
    for idir in range(3):
        x[i,idir,:] = xv[:,i,idir]
        v[i,idir,:] = xv[:,i,3+idir]
plt.figure()
for ibody in range(nb):
    plt.plot(x[ibody,0,:]/au,x[ibody,1,:]/au)
plt.xlabel('x [au]')
plt.ylabel('y [au]')
plt.axis('equal')
plt.savefig('fig_nbody_1_1.pdf')
plt.show()
