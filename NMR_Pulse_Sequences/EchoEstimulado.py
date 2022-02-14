import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

""" 
Programa - Eco Estimulado(01 /09/ 2013)

------------------------------------------------------------------------
Este programa calcula a Magnetização resultante após a aplicação da seq.
Eco Estimulado.[90 x_t1_90x_t2_90x_t_eco]
------------------------------------------------------------------------
Programa Pulso Simples - (Maio / 2020) revisão

------------------------------------------------------------------------
Este código calcula o comportamentoda Magnetização quando aplicamos um
CPMG(matricial) de NMR. 90 y_t_[180 x_t_eco]
------------------------------------------------------------------------
"""

T1 = 600
T2 = 400

x0 = 50 * 10 ** (-3)

T22 = 20

FWHM = 1 / (pi * T22)

x01 = x0 - 20 * FWHM / 2
x02 = x0 + 20 * FWHM / 2

df = np.linspace(x01, x02, 1000)

fL = T22 / (1 + ((df - x0)**2) * (2*pi*T22)**2)

theta1 = pi/2
theta2 = pi/2
T = 1000
dT = 1

Rtheta1 = np.matrix([[1, 0, 0], [0, cos(theta1), -sin(theta1)], [0, sin(theta1), cos(theta1)]])
Rtheta2 = np.matrix([[1, 0, 0], [0, cos(theta2), -sin(theta2)], [0, sin(theta2), cos(theta2)]])

E1 = np.exp(-dT/T1)
E2 = np.exp(-dT/T2)
E = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]])
B = np.matrix([0, 0, 1 - E1])

Tn1 = 150
Tn2 = 2 * Tn1

N0 = int(T / dT)
N1 = int(Tn1 / dT)
N2 = int(Tn2 / dT)

Np = 2
N3 = int(N0 - (N1 + 1) - (Np * (N2 + 1)))


M = np.zeros((N0, 3))
Ms = np.zeros((N0, 3))
M[0] = [0, 0, 1]

for f in range(0, len(df)):
    phi = 2 * pi * (df[f]) * dT
    Rphi = np.matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    N1c = 0

    M[1] = Rtheta1 @ M[0] + B
    for k in range(2, N1 + 2):
        M[k] = E @ Rphi @ M[k - 1] + B
    N1c = N1 + 1

    for n in range(1, Np + 1):
        M[N1c + 1] = Rtheta2 @ M[N1c] + B
        for k in range(1, N2 + 1):
            M[k + N1c + 1] = E @ Rphi @ M[k + N1c] + B
        N1c = N1c + N2 + 1

    for k in range(0, N3):
        M[k + N1c] = E @ Rphi @ M[k + N1c - 1] + B

    g = T22 / (1 + ((df[f] - x0)**2) * (2*pi*T22)**2)

    Ms[:, 0] = g * M[:, 0] + Ms[:, 0]
    Ms[:, 1] = g * M[:, 1] + Ms[:, 1]
    Ms[:, 2] = g * M[:, 2] + Ms[:, 2]


Ms = Ms / max(abs(Ms[0]))

tempo = np.arange(0, N0).T * dT
CurvaT2 = np.exp(-tempo / T2)


# ===== Graficando Resultados ======
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

ax1.plot(df, fL, 'k', label=f'T2* = {T22} ms')
ax1.set_title('Distribuição Lorentziana')
ax1.set_xlabel('Frequência x 10^3 Hz')
ax1.set_ylabel('Intensidade')
ax1.legend()
ax1.grid()

ax2.plot(tempo, Ms[:, 0], 'b', tempo, Ms[:, 1], 'k', tempo, Ms[:, 2], 'r', tempo, CurvaT2, 'g')
ax2.set_title('90y 90x 90x eco estimulado')
ax2.set_xlabel('Tempo (ms)')
ax2.set_ylabel('Intensisade')
ax2.grid()

fig.tight_layout(pad=1)
plt.show()
