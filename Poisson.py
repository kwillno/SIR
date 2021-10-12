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
    
# PROBLEM 2B
    
    def claim_amount_P(t, gamma, lam):
    C = 0
    X = np.random.poisson(lam*t)
    for j in range(X):
        C += np.random.exponential(1/gamma)
    return C

def claim_amount_arr(t, gamma, lam):
    X = np.random.poisson(lam*t)
    C = np.zeros(X)
    for j in range(1, X):
        C[j] = C[j-1] + np.random.exponential(1/gamma)
    return C

def total_claim_amount_P(t, gamma, lam, N):
    C = np.array([])
    for i in range(N):
        claim = claim_amount(t, gamma, lam)
        C = np.append(C, claim)
    claims_over_8mill = 0
    for elem in C:
        if elem>8:
            claims_over_8mill += 1
    P = claims_over_8mill/len(C)
    return P

def plot_claim_amounts(t, gamma, lam):
    print(f'The amount of claims exceeding 8 million: {total_claim_amount_P(t, gamma, lam, 1000)}')
    colors = ["c","m","y","k","r","g","b", "mediumspringgreen", "sandybrown", "mistyrose"]
    for i in range(10):
        exp_t = make_exp_t_arr(lam)
        exp_t_copy = exp_t.copy()
        y_val = claim_amount_arr(len(exp_t), gamma, lam)
        for k in range(len(exp_t)):
            if k>0:
                exp_t[k] += exp_t[k-1]
                #y_val[k] = y_val[k-1] + np.random.poisson(lam*exp_t_copy[k])
        for j in range(len(exp_t)-1):
            plt.hlines(y_val[j], exp_t[j], exp_t[j+1], colors[i])
            
    plt.title("Insurance Claim amounts")
    plt.xlabel("Days")
    plt.ylabel("claim amount (in millions kr)")
    plt.grid()
    plt.show()
