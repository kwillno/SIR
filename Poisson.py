import numpy as np
import matplotlib.pyplot as plt

 #this function retuns the probability of getting more than a 100 claims
def poisson_realizations_cp(lam, t, N, maxiter=False):
    
    # here we choose number of iterations. We can decide with n, or just let it be 1000
    if maxiter:
        maxiter = 1000
    else:
        maxiter = N
    
    # we will fill this array up with poisson realizations  
    realizations = np.zeros(maxiter)
    
    # i foor-loop to initialize the realizations-array
    for i in range(maxiter):
        realizations[i] = np.random.poisson(lam*t) 
        
    # this is were we compute the probability for getting more than 100 claims
    computed_probability = np.sum(realizations > 100)/len(realizations)
    return computed_probability 


# this function returns an array of poisson realizations
# it works the same as poisson_realizations_cp(), only that we return
# realizations-array instead of computed_probability
def poisson_realizations_r(lam, t, N, maxiter=False):
    if maxiter:
        maxiter = 1000
    else:
        maxiter = N
    realizations = np.zeros(maxiter)
    for i in range(maxiter):
        realizations[i] = np.random.poisson(lam*t) 
    return realizations


# problem 2a***

# this function returns an array of time intervalls where the length of each intervall 
# is decided by the exponential distribution. The sum of all the time-nitervalls stops at 59.
def make_exp_t_arr(lam):
    
    # we make an array to hold the time-intervalls
    t_vals = np.array([])
    
    # a while loop that is conditioned to stop if the sum of all the time-intervalls reach 59.
    while np.sum(t_vals)<59:
        
        # we create a time intervall
        val = np.random.exponential((1/lam), 1)[0]
        
        # and then adds that time intervall to the array
        t_vals = np.append(t_vals,val)
    return t_vals


# This function plots 10 realizations of number of claims each day from day 0 to day 59
def simulate_poisson(lam, t=59, simulation_n=10):
    
    # we need this list of colors to distinguish th plotting of the 10 poisson processes
    colors = ["c","m","y","k","r","g","b", "mediumspringgreen", "sandybrown", "mistyrose"]
    
    # a for-loop that runs 10 times. Once for each plot.
    for i in range(simulation_n):
        
        # we use make_exp_t_arr() to create an array of time-intervalls, and then makes a copy that we need later
        exp_t = make_exp_t_arr(lam)
        exp_t_copy = exp_t.copy()
        
        # we make an array of y values (number of claims) that has the same lenth and dimension as exp_t
        y_val = np.zeros(len(exp_t))
        
        # in this for loop we make the exp_t-array kumulative by summing the current and privious value for all values
        for k in range(len(exp_t)):
            if k>0:
                exp_t[k] += exp_t[k-1]
                
                # this is where we need the exp_t_copy. We need the original time-intervalls to calculaate number of claims we add to y_val[k]
                y_val[k] = y_val[k-1] + np.random.poisson(lam*exp_t_copy[k]) 
                
        # here we print  
        for j in range(len(exp_t)-1):
            plt.hlines(y_val[j], exp_t[j], exp_t[j+1], colors[i])
    plt.title("Insurance Claims")
    plt.xlabel("Days")
    plt.ylabel("#claims")
    plt.grid()
    plt.show()

simulate_poisson(1.5)
    
# PROBLEM 2B

# This function returns Z(t) and is directly implemented as such
def claim_amount_P(t, gamma, lam):
    C = 0
    X = np.random.poisson(lam*t)
    for j in range(X):
        C += np.random.exponential(1/gamma)
    return C

# this function returns an array of money-claims 
def claim_amount_arr(t, gamma, lam):
    X = np.random.poisson(lam*t)
    C = np.zeros(X)
    for j in range(1, X):
        C[j] = C[j-1] + np.random.exponential(1/gamma)
    return C

# this function returns the probability that i claim is over 8 million 
def total_claim_amount_P(t, gamma, lam, N):
    C = np.array([])
    for i in range(N):
        claim = claim_amount_P(t, gamma, lam)
        C = np.append(C, claim)
    claims_over_8mill = 0
    for elem in C:
        if elem>8:
            claims_over_8mill += 1
    P = claims_over_8mill/len(C)
    return P


# this function prints the money-claim amounts
# This function works in much the same way as simulate_poisson(), only that we use claim amounts as y values
def plot_claim_amounts(lam, t=59, gamma=10):
    print(f'The amount of claims exceeding 8 million: {total_claim_amount_P(t, gamma, lam, 1000)}')
    colors = ["c","m","y","k","r","g","b", "mediumspringgreen", "sandybrown", "mistyrose"]
    for i in range(10):
        exp_t = make_exp_t_arr(lam)
        exp_t_copy = exp_t.copy()
        y_val = claim_amount_arr(len(exp_t), gamma, lam)
        for k in range(len(exp_t)):
            if k>0:
                exp_t[k] += exp_t[k-1]
        for j in range(len(exp_t)-1):
            plt.hlines(y_val[j], exp_t[j], exp_t[j+1], colors[i])  
    plt.title("Insurance Claim amounts")
    plt.xlabel("Days")
    plt.ylabel("claim amount (in millions kr)")
    plt.grid()
    plt.show()

plot_claim_amounts(1.5)
