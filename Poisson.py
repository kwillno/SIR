import numpy as np
import matplotlib.pyplot as plt

def poisson_bool(lam, t, N):
    poisson_sum = 0
    for i in range(N+1):
        poisson_sum += (((lam*t)**i)/(np.math.factorial(i)))*np.e**(-lam*t)
    return np.random.random() < 1-poisson_sum

def poisson_realizations_maxiter(lam, t, N, maxiter):
    realizations = 0
    for i in range(maxiter):
        if poisson_bool(lam, t, N):
            realizations += 1
    computed_probability = realizations/maxiter
    return computed_probability

def simulate_poisson_t(lam, t, N):
    realizations = 0
    realization_arr = np.zeros(t)
    for i in range(t):
        if poisson_bool(lam, t, N):
            realizations += 1
            realization_arr[i] = realizations
        else:
            realization_arr[i] = realizations
    return realization_arr
    
def plot_poisson(lam, t, N):
    x_vals = np.linspace(0, 59, 59)
    real1 = simulate_poisson_t(lam, t, N)
    real2 = simulate_poisson_t(lam, t, N)
    real3 = simulate_poisson_t(lam, t, N)
    real4 = simulate_poisson_t(lam, t, N) 
    real5 = simulate_poisson_t(lam, t, N)
    real6 = simulate_poisson_t(lam, t, N)
    real7 = simulate_poisson_t(lam, t, N)
    real8 = simulate_poisson_t(lam, t, N)
    real9 = simulate_poisson_t(lam, t, N)
    real10 = simulate_poisson_t(lam, t, N)
    plt.plot(x_vals,real1)
    plt.plot(x_vals,real2)
    plt.plot(x_vals,real3)
    plt.plot(x_vals,real4)
    plt.plot(x_vals,real5)
    plt.plot(x_vals,real6)
    plt.plot(x_vals,real7)
    plt.plot(x_vals,real8)
    plt.plot(x_vals,real9)
    plt.plot(x_vals,real10)
    plt.show()
