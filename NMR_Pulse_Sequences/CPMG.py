import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

T1 = 500
T2 = 400
Fo = 25
df = np.arange(0, 50, 0.1)

theta = pi/2.0
theta2 = pi
T = 2000
dT = 1

T2inom = 100
T22 = 1/(1/float(T2) + (1/float(T2inom)))

Rtheta = np.matrix([[1, 0, 0], [0, cos(theta), sin(theta)], [0, -sin(theta), cos(theta)]])
Rtheta2 = np.matrix([[cos(theta2), 0, sin(theta2)], [0, 1, 0], [-sin(theta2), 0, cos(theta2)]])
E1 = np.exp(-1*dT/float(T1))
E2 = np.exp(-1*dT/float(T2))
E = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]])
B = np.matrix([0, 0, 1 - E1])

Tn1 = 200
Tn2 = 2*Tn1
Np = 4

N0 = int(T/float(dT))
N1 = int(Tn1/float(dT))
Nlc = N1
N2 = int(Tn2/float(dT))
N3 = N0 - (N1+(Np - 1)*N2)

M = np.zeros((N0, 3))
Ms = np.zeros((N0, 3))

M[0] = [0, 0, 1]

for f in range(0, len(df)):
    phi = 2*pi*(df[f])*dT/1000.0
    Rphi = np.matrix([[cos(phi), -1*sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    M[1] = Rtheta @ M[0] + B

    for k in range(2, Nlc + 1):
        M[k] = E @ Rphi @ M[k - 1] + B

    for n in range(0, Np - 1):
        M[Nlc + 1] = Rtheta2 @ M[Nlc] + B
        for k in range(1, N2 - 1):
            M[k + Nlc + 1] = E @ Rphi @ M[k + Nlc] + B
        Nlc += N2 - 1

    M[Nlc + 1] = Rtheta2 @ M[Nlc] + B
    for k in range(2, N3 - 1):
        M[k + Nlc] = E @ Rphi @ M[k + Nlc - 1] + B

    Nlc = N1

    g = T22 / ((1 + (df[f] - Fo)**2) * (2*pi*T22/1000)**2)

    Ms[:, 0] = g*M[:, 0] + Ms[:, 0]
    Ms[:, 1] = g*M[:, 1] + Ms[:, 1]
    Ms[:, 2] = g*M[:, 2] + Ms[:, 2]

Ms = Ms/max(Ms[0])

tempo = np.arange(0, N0).T * dT
CurvaT2 = [np.exp(-t/T2) for t in tempo]
CurvaT22 = [np.exp(-t/T22) for t in tempo]

# ===== Graficando Resultados =====

plt.plot(tempo, CurvaT2, 'g')
# plt.plot(tempo, CurvaT22, 'y')
plt.plot(tempo, Ms[:, 0], 'b')
plt.plot(tempo, Ms[:, 1], 'k')
plt.plot(tempo, Ms[:, 2], 'r')
plt.grid()

plt.show()
