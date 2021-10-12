import numpy as np
import matplotlib.pyplot as plt

def poisson_realizations_cp(lam, t, N, maxiter=False):
    if maxiter:
        maxiter = 1000
    else:
        maxiter = N
    realizations = np.zeros(maxiter)
    for i in range(maxiter):
        realizations[i] = np.random.poisson(lam*t) 
    computed_probability = np.sum(realizations > 100)/len(realizations)
    return computed_probability 

def poisson_realizations_r(lam, t, N, maxiter=False):
    if maxiter:
        maxiter = 1000
    else:
        maxiter = N
    realizations = np.zeros(maxiter)
    for i in range(maxiter):
        realizations[i] = np.random.poisson(lam*t) 
    computed_probability = np.sum(realizations > 100)/len(realizations)
    return realizations


# problem 2a***

def make_exp_t_arr(lam):
    t_vals = np.array([])
    while np.sum(t_vals)<59:
        val = np.random.exponential((1/lam), 1)[0]
        t_vals = np.append(t_vals,val)
    return t_vals

def simulate_poisson(lam, t, N, simulation_n=10):
    colors = ["c","m","y","k","r","g","b", "mediumspringgreen", "sandybrown", "mistyrose"]
    for i in range(simulation_n):
        #y_val = np.array([0, 1])
        exp_t = make_exp_t_arr(lam)
        exp_t_copy = exp_t.copy()
        y_val = np.zeros(len(exp_t))
        for k in range(len(exp_t)):
            if k>0:
                exp_t[k] += exp_t[k-1]
                y_val[k] = y_val[k-1] + np.random.poisson(lam*exp_t_copy[k]) 
        #print(exp_t_copy)
        #print(y_val)
        for j in range(len(exp_t)-1):
            plt.hlines(y_val[j], exp_t[j], exp_t[j+1], colors[i])
    plt.title("Insurance Claims")
    plt.xlabel("Days")
    plt.ylabel("#claims")
    plt.grid()
    plt.show()
