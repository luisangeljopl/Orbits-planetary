import numpy as np
import scipy as sp 
import scipy.integrate as integrate
import matplotlib.pylab as plt
import math
import pylab

# Konstanten 1 Tag [s] und 1 Jahr [s].
tag = 24 * 60 * 60
jahr = 365.25 * tag
# Eine Astronomische Einheit [m].
AE = 1.495978707e11
# Skalierungsfaktor für die Darstellung der Beschleunigung
# [AE / (m/s2)] und Geschwindigkeit [AE / (m/s)].
scal_a = 20
scal_v = 1e-5

# Simulationsdauer T und dargestellte Schrittweite dt [s].
T = 1 * jahr
dt = 1 * tag

# Masse der Sonne M [kg].
M = 1.9885e30
# Gravitationskonstante [m3 / (kg * s2)].
G = 6.674e-11
# Anfangsposition des Planeten [m].
r0 = np.array([152.10e9, 0.0])
# Anfangsgeschwindigkeit des Planeten [m/s].
v0 = np.array([0.0, 29.29e3])

def dgl(t, u):
    r, v = np.split(u, 2)
    a = - G * M * r / np.linalg.norm(r) ** 3
    return np.concatenate([v, a])

# Lege den Zustandsvektor zum Zeitpunkt t=0 fest.
u0 = np.concatenate((r0, v0))
# Löse die Bewegungsgleichung.
result = scipy.integrate.solve_ivp(dgl, [0, T], u0, dense_output=True)
t_s = result.t
r_s, v_s = np.split(result.y, 2)
# Berechne die Interpolation auf einem feinen Raster.
t = np.arange(0, np.max(t_s), dt)
r, v = np.split(result.sol(t), 2)

def update(n):
    # Aktualisiere die Position des Himmelskörpers.
    planet.set_data(r[:, n] / AE)
    # Berechne die Momentanbeschleunigung und aktualisiere die
    # Vektorpfeile.
    u = np.concatenate((r[:, n], v[:, n]))
    u_punkt = dgl(t[n], u)
    a = np.split(u_punkt, 2)[1]

    arrow_a.set_positions(r[:, n] / AE,
                      r[:, n] / AE + scal_a * a)
                      
    arrow_v.set_positions(r[:, n] / AE,
                      r[:, n] / AE + scal_v * v[:, n])


    # Aktualisiere das Textfeld für die Zeit.
    text_t.set_text(f't = {t[n] / tag:.0f} d')                     
    return planet, arrow_v, arrow_a, text_t 
