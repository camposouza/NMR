import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

T1 = 600
T2 = 400
Fo = 15
df = np.arange(0, 50, 0.1)

theta = pi
theta2 = pi/2.0
T = 1600
dT = 1

T2inom = 200
T22 = 1/(1/float(T2) + (1/float(T2inom)))

Rtheta = np.matrix([[1, 0, 0], [0, cos(theta), sin(theta)], [0, -sin(theta), cos(theta)]])
Rtheta2 = np.matrix([[1, 0, 0], [0, cos(theta2), sin(theta2)], [0, -sin(theta2), cos(theta2)]])
E1 = np.exp(-dT/float(T1))
E2 = np.exp(-dT/float(T2))
E = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]])
B = np.matrix([0, 0, 1 - E1])

Tn1 = 100

N0 = int(T/float(dT))
N1 = int(Tn1/float(dT))
N2 = N0 - N1

M = np.zeros((N0, 3))
Ms = np.zeros((N0, 3))

M[0] = [0, 0, 1]

for f in range(0, len(df)):
    phi = 2*pi*(df[f])*dT/1000.0
    Rphi = np.matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    M[1] = Rtheta @ M[0] + B

    for k in range(2, N1 + 1):
        M[k] = E @ Rphi @ M[k - 1] + B

    M[N1 + 1] = Rtheta2 @ M[N1] + B

    for k in range(1, N2 - 1):
        M[k + N1 + 1] = E @ Rphi @ M[k + N1] + B

    g = T22 / ((1 + (df[f] - Fo)**2) * (2*pi*T22/1000)**2)

    Ms[:, 0] = g*M[:, 0] + Ms[:, 0]
    Ms[:, 1] = g*M[:, 1] + Ms[:, 1]
    Ms[:, 2] = g*M[:, 2] + Ms[:, 2]

Ms = Ms/max(Ms[0])

tempo = np.arange(0, N0).T * dT

# ===== Graficando Resultados =====

# plt.plot(tempo, CurvaT22, 'y')
plt.plot(tempo, Ms[:, 0], 'b')
plt.plot(tempo, Ms[:, 1], 'k')
plt.plot(tempo, Ms[:, 2], 'r')
plt.grid()

plt.show()
