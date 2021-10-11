import numpy as np
import matplotlib.pyplot as plt

def poisson_realizations(lam, t, N, maxiter):
    realizations = np.zeros(maxiter)
    for i in range(maxiter):
        realizations[i] = np.random.poisson(lam*t, 1) 
    computed_probability = np.sum(realizations > 100)/len(realizations)
    return computed_probability, realizations         


# problem 2a***

def make_exp_t_arr(lam):
    t_vals = np.array([])
    while np.sum(t_vals)<59:
        val = np.random.exponential(lam, 1)[0]
        t_vals = np.append(t_vals,val)
    return t_vals

def simulate_poisson(lam, t, N, simulation_n=10):
    for i in range(simulation_n):
        y_val = 0
        #a, poisson_vals = poisson_realizations(lam, t, N, )
        exp_t = make_exp_t_arr(lam)
        y_val = np.random.poisson(lam*0, 1)
        for j in range(len(exp_t)):
            plt.plot(make_exp_t_arr[j], y_val, '_', 'hline')
            y_val = np.random.poisson(lam*(i+1), 1)
        plt.show()
