
import matplotlib.pyplot as plt
import scipy.integrate as spi
import numpy as np
    
#Gravitational constant
G = 4*np.pi**2 

#Mass of the star
m_star = 1

#Final time value, plot the solution for t in [0,tf]
tf = 2 

#Initial conditions: [x position, x speed, y position, y speed]
u0 = [1,0,0,6.5]
def odefun(u,t):

    #dudt = [x, x', y, y']
    dudt = [0,0,0,0] 
    D3 = (u[0]**2 + u[2]**2)**(3/2)
    
    dudt[0] = u[1]
    dudt[1] = -G*m_star*u[0]/D3
    dudt[2] = u[3]
    dudt[3] = -G*m_star*u[2]/D3
    return dudt
t = np.linspace(0,tf,1000)
u = spi.odeint(odefun,u0,t)
plt.plot(u[:,0],u[:,2],0,0,'ro')
plt.axis('equal')
plt.grid(True)
plt.show()
