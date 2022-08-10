import numpy as np
from numpy import pi, cos, sin, exp
import matplotlib.pyplot as plt
from scipy import fftpack

T1 = 600
T2 = 400

x0 = 20 * 10 ** (-3)

T22 = 200

FWHM = 1 / (np.pi * T22)

x01 = x0 - 20 * FWHM / 2
x02 = x0 + 20 * FWHM / 2

df = np.linspace(x01, x02, 1000)

fL = T22 / (1 + ((df - x0) ** 2) * (2 * pi * T22) ** 2)

T = 150
dT = 0.2

E1 = np.exp(-1 * dT / float(T1))
E2 = np.exp(-1 * dT / float(T2))
E = np.array([[E2, 0, 0], [0, E2, 0], [0, 0, E1]])
B = np.array([[0], [0], [1 - E1]])

N0 = int(T / float(dT))
M0 = np.matrix([[0], [0], [1]])

alpha = np.linspace(0, 2 * np.pi, 20)

Signalfft = np.zeros((len(alpha), N0))

for alp in range(0, len(alpha)):

    M = np.zeros((3, N0))
    Ms = np.zeros((3, N0))

    # Rflip = np.matrix([[1, 0, 0], [0, cos(alpha[alp]), -sin(alpha[alp])], [0, sin(alpha[alp]), cos(alpha[alp])]])  # pulsos em x
    Rflip = np.array([[cos(alpha[alp]), 0, sin(alpha[alp])], [0, 1, 0], [-sin(alpha[alp]), 0, cos(alpha[alp])]])

    for f in range(0, len(df)):
        phi = 2 * np.pi * (df[f]) * dT
        Rphi = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        M[:, 0] = (np.dot(Rflip, M0 + B)).reshape((1, 3))  # .reshape((3, 1)))

        for k in range(1, N0):
            M[:, k] = (np.dot(np.dot(E, Rphi), (M[:, k - 1]).reshape((3, 1))) + B).reshape(1, 3)

        g = T22 / float((1 + ((df[f] - x0) ** 2) * (2 * np.pi * T22) ** 2))

        Ms[0, :] = g * M[0, :] + Ms[0, :]
        Ms[1, :] = g * M[1, :] + Ms[1, :]
        Ms[2, :] = g * M[2, :] + Ms[2, :]

    y = np.array(Ms[1, :] + 1j * Ms[2, :])

    Y = fftpack.fft(y)
    Y_shift = fftpack.fftshift(Y)

    Signalfft[alp, :] = np.real(Y_shift[None, :])

dt = dT

f_k = fftpack.fftfreq(y.size, d=dt)
f = fftpack.fftshift(f_k)


f = f * 1000  # [Hz]

tempo = np.arange(0, N0) * dT


maximos = np.zeros(len(alpha))
for i in range(0, len(alpha)):
    max_ind = np.argmax(np.abs(Signalfft[i, :]))

    maximos[i] = Signalfft[i, max_ind]


# ===== Plot Resultados ======
fig = plt.figure()

ax1 = fig.add_subplot(3, 1, 1, )
ax1.plot(tempo, Ms[0, :], 'b', tempo, Ms[1, :], 'k')
ax1.set_title("Sinal decaimento")
ax1.set_xlabel("tempo")
ax1.set_ylabel("Intensidade")

ax2 = fig.add_subplot(3, 1, 2, )
ax2.plot(f, Signalfft[0, :])
ax2.set_title("Transformada Fourrier")
ax2.set_xlabel("Frequência Hz")
ax2.set_ylabel("Intensidade")

ax3 = fig.add_subplot(3, 1, 3, )
ax3.plot(alpha, maximos)
ax3.set_title("Calibração de Pulso")
ax3.set_xlabel("Ângulo de Puls(rad)")
ax3.set_ylabel("Intensidade")

fig.tight_layout()
plt.show()
