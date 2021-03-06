import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.optimize
import pandas as pd


def main():
    # Reading results
    data = pd.read_csv('std_data.txt', delimiter='\t', index_col=0, header=None)
    t = np.array(data.index)

    results = pd.DataFrame(index=['A1', 'K1', 'A2', 'K2', 'C'])
    for i in range(0, data.shape[1]):
        y = np.array(data.iloc[:, 0])

        # Non-linear Fit
        A1, K1, A2, K2, C = fit_exp_nonlinear(t, y)
        fit_y = model_func(t, A1, K1, A2, K2, C)

        # Graphing Result
        if i == 0:
            fig = plt.figure()
            ax1 = fig.add_subplot()

            ax1.scatter(t, y, c='red', s=1,
                        label='Dados experimentais')
            ax1.plot(t, fit_y, 'b-',
                     label='Curva encontrada:\n $y = %0.2f e^{%0.2f t} + %0.2f e^{%0.2f t} + %0.2f$'
                           % (A1, K1, A2, K2, C))
            ax1.legend(bbox_to_anchor=(1.05, 1.1), fancybox=True, shadow=True)

        # Filling the Sheet
        results.insert(i, i, [A1, K1, A2, K2, C])

    results.to_csv('BiFitting', sep='\t')
    plt.show()


def model_func(t, A1, K1, A2, K2, C):
    return A1 * np.exp((-1/K1) * t) + A2 * np.exp((-1/K2) * t) + C


def fit_exp_nonlinear(t, y):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, maxfev=1000)
    A1, K1, A2, K2, C = opt_parms
    return A1, K1, A2, K2, C


if __name__ == '__main__':
    main()
