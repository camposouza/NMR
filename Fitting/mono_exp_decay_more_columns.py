import numpy as np
import scipy as sp
import scipy.optimize
import pandas as pd


def main():
    # Reading results
    # todo verificar como os dados são passados. São sempre com índice à esquerda e sem cabeçalho?
    data = pd.read_table('data.txt', index_col=0, header=None)
    t = np.array(data.index)
    for i in range(0, data.shape[1]):
        y = np.array(data.iloc[:, i])

        # Non-linear Fit
        A, K, C = fit_exp_nonlinear(t, y)
        fit_y = model_func(t, A, K, C)

        # Showing Results
        print(f'Conjunto de dados {i+1}:\n'
              f'  A = {A:.2f}\n  K = {K:.2f}\n  C = {C:.2f}')


def model_func(t, A, K, C):
    return A * np.exp(-K * t) + C


def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C


if __name__ == '__main__':
        main()