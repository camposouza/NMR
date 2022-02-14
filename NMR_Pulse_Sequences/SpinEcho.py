import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

T1 = 500
T2 = 400
Fo = 15
df = np.arange(-50, 50, 0.1)

theta = pi/2.0
theta2 = pi
T = 1500
dT = 1

T2inom = 100
T22 = 1/(1/float(T2) + (1/float(T2inom)))

Rtheta = np.matrix([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-1*np.sin(theta), 0, np.cos(theta)]])
Rtheta2 = np.matrix([[1, 0, 0], [0, np.cos(theta2), np.sin(theta2)], [0, -1*np.sin(theta2), np.cos(theta2)]])
E1 = np.exp(-1*dT/float(T1))
E2 = np.exp(-1*dT/float(T2))
E = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]])
B = np.matrix([0, 0, 1 - E1])

Tn1 = 400

N0 = int(T/float(dT))
N1 = int(Tn1/float(dT))
N2 = N0 - N1

M = np.zeros((N0, 3))
Ms = np.zeros((N0, 3))

M[0] = [0, 0, 1]

for f in range(0, len(df)):
    phi = 2*pi*(df[f])*dT/1000.0
    Rphi = np.matrix([[np.cos(phi), -1*np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
    M[1] = Rtheta @ M[0] + B

    for k in range(2, N1 + 1):
        M[k] = E @ Rphi @ M[k - 1] + B

    M[N1 + 1] = Rtheta2 @ M[N1] + B
    for k in range(2, N2 - 1):
        M[k + N1] = E @ Rphi @  M[k + N1 - 1] + B

    g = T22 / ((1 + (df[f] - Fo)**2) * (2*pi*T22/1000)**2)

    Ms[:, 0] = g*M[:, 0] + Ms[:, 0]
    Ms[:, 1] = g*M[:, 1] + Ms[:, 1]
    Ms[:, 2] = g*M[:, 2] + Ms[:, 2]

Ms = Ms/max(Ms[0])

tempo = np.arange(0, N0).T * dT
CurvaT2 = [np.exp(-t/float(T2)) for t in tempo]
CurvaT22 = [np.exp(-t/float(T22)) for t in tempo]

# ===== Graficando Resultados =====

plt.plot(tempo, CurvaT2, 'g')
# plt.plot(tempo, CurvaT22, 'y')
plt.plot(tempo, Ms[:, 0], 'b')
plt.plot(tempo, Ms[:, 1], 'k')
plt.plot(tempo, Ms[:, 2], 'r')
plt.grid()

plt.show()