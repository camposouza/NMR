import numpy as np
from numpy import pi, sin, cos
import matplotlib.pyplot as plt

T1 = 250
T2 = 200

x0 = 5000 * (10 ** -3)

T22 = 0.500

FWHM = 1 / (pi * T22)

x01 = x0 - 10 * FWHM / 2
x02 = x0 + 10 * FWHM / 2

df = np.linspace(x01, x02, 1000)

fL = T22 / (1 + ((df - x0) ** 2) * (2 * pi * T22) ** 2)

thetaCP = pi / 2
theta1 = pi / 2
theta2 = pi / 2
T = 300
dT = 0.05

Tp = 0.3

# RthetaCP = np.matrix([ [cos(thetaCP), 0, sin(thetaCP)], [0, 1, 0], [-sin(thetaCP), 0, cos(thetaCP)] ])  # Fase Y
RthetaCP = np.matrix([[1, 0, 0], [0, cos(thetaCP), -sin(thetaCP)], [0, sin(thetaCP), cos(thetaCP)]])  # Fase x

Rtheta1 = np.matrix([[1, 0, 0], [0, cos(theta1), -sin(theta1)], [0, sin(theta1), cos(theta1)]])
Rtheta2 = np.matrix([[1, 0, 0], [0, cos(theta2), -sin(theta2)], [0, sin(theta2), cos(theta2)]])

E1 = np.exp(-1*dT/float(T1))
E2 = np.exp(-dT/T2)
E = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]])
B = np.matrix([[0], [0], [1 - E1]])

Tn2 = 0.30
Tn1 = 0.30

N0 = round(T / dT)
N1 = round(Tn1 / dT)
N2 = round(Tn2 / dT)

Np = np.floor((N0 - (N1 + 1)) / (N2 + 1))
N3 = N0 - (N1 + 1) - (Np * (N2 + 1))

M = np.zeros((3, N0))
Ms = np.zeros((3, N0))
M0 = np.matrix([[0], [0], [1]])

for f in range(0, len(df)):
    phi = 2 * pi * df[f] * dT
    Rphi = np.matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])

    M[:, 1] = (np.dot(Rtheta1, M[:, 0].reshape((3, 1))) + B).reshape((1, 3))
    for k in range(2, N1 + 2):
        M[:, k] = (np.dot(np.dot(E, Rphi), M[:, k - 1].reshape((3, 1))) + B).reshape((1, 3))
    N1c = N1 + 1

    for n in range(1, int(np.floor(Np / 2))):
        M[:, N1c] = (np.dot(Rtheta1, M[:, N1c - 1].reshape((3, 1))) + B).reshape((1, 3))
        for k in range(0, N2):
            M[:, k + N1c + 1] = (np.dot(np.dot(E, Rphi), M[:, k + N1c].reshape((3, 1))) + B).reshape((1, 3))
        N1c = N1c + N2 + 1

        M[:, N1c] = (np.dot(Rtheta2, M[:, N1c - 1].reshape((3, 1))) + B).reshape((1, 3))
        for k in range(0, N2):
            M[:, k + N1c + 1] = (np.dot(np.dot(E, Rphi), M[:, k + N1c].reshape((3, 1))) + B).reshape((1, 3))
        N1c = N1c + N2 + 1

    for k in range(0, int(N3 - 1)):
        M[:, k + N1c + 1] = (np.dot(np.dot(E, Rphi), M[:, k + N1c].reshape((3, 1))) + B).reshape((1, 3))

    g = T22 / (1 + ((df[f] - x0) ** 2) * (2 * pi * T22) ** 2)

    Ms[0, :] = g * M[0, :] + Ms[0, :]
    Ms[1, :] = g * M[1, :] + Ms[1, :]
    Ms[2, :] = g * M[2, :] + Ms[2, :]


Mod = ((Ms[1, :]) ** 2 + (Ms[2, :]) ** 2) ** 0.5

Ms = Ms / max(Mod)
Mod = Mod / max(Mod)

tempo = np.arange(0, N0) * dT

SigMod = [(x - N3 - N2 - 1) for x in Mod]
time = [(x - N3 - N2 - 1) for x in tempo]


# ===== Graficando Resultados ======
fig = plt.figure()

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(df, fL, 'k', label=f'T2 *= {T22} ms')
ax1.set_title('Distribuicao Lorentziana')
ax1.set_xlabel('Frequência (x10^3 Hz)')
ax1.set_ylabel('Intensidade')
ax1.grid()

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(tempo, Ms[0, :], 'b-', tempo, Ms[1, :], 'k-', tempo, Ms[2, :], 'r--', time, SigMod, 'o')
ax2.set_title('CPCWFP')
ax2.set_xlabel('Tempo (ms)')
ax2.set_ylabel('Intensidade')
ax2.grid()

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(tempo, Mod, 'k') # 'b-', tempo, Ms[1, :], 'k-', tempo, Ms[2, :], 'r--');
ax3.set_title('Módulo Magnetização')
ax3.set_xlabel('Tempo (ms)')
ax3.set_ylabel('Intensidade')
ax3.grid()

# ax4 = fig.add_subplot(2, 2, 4)
# ax4.plot(time, SigMod, 'o', xfit, yfit, 'r',  # , 'b-', tempo, Ms[1, :], 'k-', tempo, Ms[2, :], 'r--',
#              label=f'T* = {Tstar} ')
# ax4.set_title('Módulo Magnetização')
# ax4.set_ylim([-1, 1])
# ax4.set_xlabel('Tempo (ms)')
# ax4.set_ylabel('Intensidade')
# # text(1000, 0.4, ['T*=', num2str(Tstar), ' ms'])
# # text(1000, 0.3, ['T1=', num2str(Tstar1), ' ms'])
# # text(1000, 0.2, ['T2=', num2str(Tstar2), ' ms'])
# ax4.grid()

fig.tight_layout()
plt.show()
