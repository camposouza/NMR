import numpy as np
from numpy import pi, cos, sin
import matplotlib.pyplot as plt
from scipy import fftpack
"""
------------------------------------------------------------------------
           Simulacao Imaging NMR               tiagobuemoraes@gmail.com
------------------------------------------------------------------------
"""


T1 = 300
T2 = 100

Np = 1
Tp1 = 10
Tp2 = 1
alpha = pi/2
alpha2 = pi

x0 = 0
df = 0

T2inom = 99.994
T22 = 1 / ((1 / T2) + (1 / T2inom))

T = 100
dT = 0.1

N0 = int(T/dT)
N1 = int(Tp1/dT)
N2 = int(Tp2/dT)
N4 = int(N0 - (Np * (3*N2 + 2*N1 + 4)))
N1c = int(0)

# ===== Matriz Rotação ====
Rpx = np.matrix([[1, 0, 0], [0, cos(alpha), sin(alpha)], [0, -sin(alpha), cos(alpha)]])  # pulsos em x
Rpmx = np.matrix([[1, 0, 0], [0, cos(alpha2), sin(alpha2)], [0, -sin(alpha2), cos(alpha2)]])  # pulsos em -x
Rpy = np.matrix([[cos(alpha), 0, sin(alpha)], [0, 1, 0], [-sin(alpha), 0, cos(alpha)]])  # pulsos em y
Rpmy = np.matrix([[cos(-alpha2), 0, sin(-alpha2)], [0, 1, 0], [-sin(-alpha2), 0, cos(-alpha2)]])  # pulsos em y

E1 = np.exp(-1*dT/float(T1))
E2 = np.exp(-1*dT/float(T2))
B = np.matrix([0, 0, 1 - E1])

M = np.zeros((N0, 3))
Ms = np.zeros((N0, 3))
Mag = np.zeros((N0, 3))
M[1] = [0, 0, 1]

# discretizacao espaco Z                [ -1 cm --- +1cm ]
z = [i/10000 for i in range(-200, 201)]
xL = np.linspace(min(z), max(z), len(z))

# Objeto 2
g = np.zeros(len(z))
for ob in range(0, len(z)):
    if (ob + 1) < (len(z) / 4):
        g[ob] = 1
    elif len(z) / 4 < (ob + 1) < len(z) / 2:
        g[ob] = 2
    else:
        g[ob] = 0

# # Objeto 3
# g = np.zeros(len(z))
# for ob in range(0, len(z)):
#     g[ob] = exp(-z[ob] * 300)


for fz in range(0, len(z)):
    gamma = 42576000
    Grad = 1 / (0.01 * (Tp2 / 1000) * gamma)

    Phg = Grad * z[fz] * gamma * (Tp2 / 1000)

    phi = 0
    Rz = np.matrix([[cos(phi), -sin(phi), 0], [sin(phi), cos(phi), 0], [0, 0, 1]])
    A = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]]) @  Rz

    phig = phi + Phg
    Rgrad = np.matrix([[cos(phig), -sin(phig), 0], [sin(phig), cos(phig), 0], [0, 0, 1]])
    Agr = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]]) @ Rgrad

    phig2 = phi - Phg
    Rgrad2 = np.matrix([[cos(phig2), -sin(phig2), 0], [sin(phig2), cos(phig2), 0], [0, 0, 1]])
    Agr2 = np.matrix([[E2, 0, 0], [0, E2, 0], [0, 0, E1]]) @ Rgrad2

    M[N1c + 2] = A @ Rpx @ M[N1c + 1] + B # Pulso  90ºx

    for k in range(2, N1 + 1):
        M[k + N1c + 1] = A @ M[k + N1c] + B
    N1c = N1c + N1

    for n in range(1, Np + 1):
        for k in range(1, N2 + 1):
            M[k + N1c + 1] = Agr @ M[k + N1c] + B
        N1c = N1c + N2

        for k in range(1, N1 + 1):
            M[k + N1c + 1] = A @ M[k + N1c] + B
        N1c = N1c + N1

        for k in range(1, (2 * N2) + 1):
            M[k + N1c + 1] = Agr2 @ M[k + N1c] + B
        N1c = N1c + 2 * N2

    for k in range(1, N4 + 1):
        M[k + N1c + 1] = A @ M[k + N1c] + B
    N1c = 0

    Ms[:, 0] = Ms[:, 0] + g[fz] * M[:, 0]
    Ms[:, 1] = Ms[:, 1] + g[fz] * M[:, 1]
    Ms[:, 2] = Ms[:, 2] + g[fz] * M[:, 2]


Ms = Ms / (max(max(Ms[:, 0]), max(Ms[:, 1])))

Mod = (Ms[:, 0] ** 2 + (Ms[:, 1]) ** 2) ** 0.5

# Transformada de Fourier
y = Ms[2 * N1 + N2 + 1:2 * N1 + 3 * N2, 1] + 1j * Ms[2 * N1 + N2 + 1: 2 * N1 + 3 * N2, 0]
N = len(y)

t = [num * dT/1000 for num in range(0, N)]  # Time vector  (seg)
SW = 1 / (dT/1000)

# Transformada de Fourier
Y = fftpack.fft(y)
Y_shift = fftpack.fftshift(Y)
Signalfft = abs(Y_shift)

# Calcula eixo das frequências
f_k = fftpack.fftfreq(y.size, d=dT)
f = fftpack.fftshift(f_k)

dim = (2*pi*f / ((Tp2/dT) * Grad * gamma))
fL = g

# ===== Graficando Resultados ======
fig1 = plt.figure()

# Objeto
ax1 = fig1.add_subplot(3, 1, 1)
ax1.plot(xL, fL, 'k')
ax1.set_xlim(-0.04, 0.04)
ax1.set_ylim(0, 2.1)
ax1.set_title('Objeto')
ax1.set_xlabel('d (cm)')
ax1.set_ylabel('Intensidade')
ax1.grid()

# Echo
ax2 = fig1.add_subplot(3, 1, 2)
tempo = np.arange(0, N0) * dT
CurvaT2 = np.exp(-tempo / T2)
ax2.plot(tempo, Ms[:, 0], 'b', tempo, Ms[:, 1], 'k-')
ax2.set_title('Imaging NMR')
ax2.set_xlabel('Tempo (ms)')
ax2.set_ylabel('Intensidade')
ax2.grid()

# Módulo Magnetização
ax3 = fig1.add_subplot(3, 1, 3)
ax3.plot(tempo, Mod, 'k', tempo, CurvaT2, 'g--')
ax3.set_title('Módulo Magnetização')
ax3.set_xlabel('Tempo (ms)')
ax3.set_ylabel('Intensidade')
ax3.grid(axis='both')


fig2 = plt.figure()

# Echo
ax1_2 = fig2.add_subplot(3, 1, 1)
ax1_2.plot(t, np.real(y), t, np.imag(y))
ax1_2.set_title('Echo')
ax1_2.set_xlabel('Tempo (ms)')
ax1_2.set_ylabel('Intensidade')

# Transformada Fourier
ax2_2 = fig2.add_subplot(3, 1, 2)
ax2_2.plot(f, Signalfft)
ax2_2.set_title('Transformada Fourier')
ax2_2.set_xlabel('Frequência Hz')
ax2_2.set_ylabel('Intensidade')

# Imagem
ax3_2 = fig2.add_subplot(3, 1, 3)
ax3_2.plot(dim, Signalfft)
ax3_2.set_title('Imagem')
ax3_2.set_xlabel('d (cm)')
ax3_2.set_ylabel('Intensidade')

fig1.tight_layout()
fig2.tight_layout()

plt.show()
