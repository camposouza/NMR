from matplotlib import pyplot as plt
from numpy import exp, log10
import numpy as np

"""
------------------------------------------------------------------------
ToolBox Laplace - (29 / 04 / 2014)
Tiago Bueno de Moraes
------------------------------------------------------------------------
"""
alpha = 10
dmin = 0
interacoes = 15
pontos = 200

data = open('std_data.txt', 'r')

Sig = []
for row in data:
    Sig.append([float(x) for x in row.split()])
Sig = np.matrix(Sig)
data.close()

t = np.matrix(Sig[:, 0] / 1000)
Mx = np.matrix(Sig[:, 1])
My = np.zeros(np.size(Mx))

n = len(Mx)
Ti = 0.001
Tf = 10

T = np.logspace(log10(Ti), log10(Tf), pontos)

K = []
for i in range(0, n):
    row = []
    for j in range(0, pontos):
        row.append(exp(-t[i] / T[j]))
    K.append(row)
K = np.matrix(K)

O = K.conj().T @ K

VV = np.zeros((len(O), len(O)))

for k in range(1, len(O)-1):
    VVa = np.zeros((len(O), len(O)))

    VVa[k, k] = 4
    VVa[k - 1, k - 1] = 1
    VVa[k + 1, k + 1] = 1
    VVa[k + 1, k - 1] = 1
    VVa[k - 1, k + 1] = 1
    VVa[k, k + 1] = -2
    VVa[k, k - 1] = -2
    VVa[k - 1, k] = -2
    VVa[k + 1, k] = -2

    VV = VV + VVa


VV[0, 0] = 10000000
VV[len(O)-1, len(O)-1] = 10000000

L = alpha * VV


g = np.zeros((1, pontos))


for inte in range(1, interacoes + 1):
    Soma = O + L

    U, sDiag, Vh = np.linalg.svd(Soma)
    S = np.zeros(Soma.shape)
    np.fill_diagonal(S, sDiag)
    V = Vh.T.conj()

    vd = np.diag(S)
    a = len(vd)
    vd = vd.copy().reshape(200, 1)
    vd[vd <= dmin] = []
    np.fill_diagonal(S, vd)

    for j in range(a, len(vd)+1, -1):
        U[:, j] = []
        V[:, j] = []

    invSoma = U @ (np.linalg.inv(S)) @ V.conj().T

    g = Mx.conj().T @ K @ invSoma

    indf = np.nonzero(g < 0)
    for i in range(0, len(indf)):
        L[indf[i], indf[i]] = L[indf[i], indf[i]] + 10000


g = g / np.sum(g)


# ===== Graficando Resultados =====
fig1 = plt.figure()

ax1 = fig1.add_subplot(2, 2, 1)
ax1.plot(t, Mx, t, My)
ax1.set_title('CPMG')

ax2 = fig1.add_subplot(2, 2, 2)
ax2.stem(range(1, len(S) + 1), vd)
ax2.set_title('Valores Singulares')
ax2.set_yscale('log')

ax3 = fig1.add_subplot(2, 1, 2)
ax3.plot(T, g)
ax3.set_title('Distribuição T2')
ax3.set_xlim(Ti, Tf)
ax3.set_xscale('log')

fig1.tight_layout()
plt.show()
