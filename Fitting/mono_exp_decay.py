import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import pandas as pd


def main():
    # Reading results
    data = pd.read_table('data.txt', index_col=0, header=None)
    t = np.array(data.index)
    for i in range(0, data.shape[1]):
        y = np.array(data.iloc[:, 0])

        # Non-linear Fit
        A, K, C = fit_exp_nonlinear(t, y)
        fit_y = model_func(t, A, K, C)

        # Graphing Result
        if i == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot()

            ax1.scatter(t, y, c='red', s=1,
                        label='Dados experimentais')
            ax1.plot(t, fit_y, 'b-',
                     label='Curva encontrada:\n $y = %0.2f e^{%0.2f t} + %0.2f$' % (A, K, C))
            ax1.legend(bbox_to_anchor=(1.05, 1.1), fancybox=True, shadow=True)

        # Showing Results
        print(f'Conjunto de dados {i + 1}:\n'
              f'  A = {A:.2f}\n  K = {K:.2f}\n  C = {C:.2f}')

    plt.show()


def model_func(t, A, K, C):
    return A * np.exp((-1/K) * t) + C


def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A, K, C = opt_parms
    return A, K, C


if __name__ == '__main__':
    main()
