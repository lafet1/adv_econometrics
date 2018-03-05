#%%
import numpy as np
from numpy import zeros, random, sqrt, mean, percentile
from numpy import transpose as t
from numpy.linalg import inv
from matplotlib import pyplot as plt
from scipy import stats
a_list = [1, 0.6, 0.3, 0.15, 0.07, 0.04, 0.02, 0]
rho_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
N=100
k=10
beta = 0
REP = 1000

#%%
'''
EXERCISE 1
'''

Z = random.normal(size = (N,k))

results = zeros((len(a_list), len(rho_list)))
counter_a = 0
for a in a_list:
    PI = zeros(shape=k)
    PI[0] = a
    counter_rho = 0
    for rho in rho_list:
        SIGMA = np.array([[1, rho], [rho, 1]])
        tstat_result = zeros(REP)
        random.seed(2110)

        for i in range(REP):
            
            eps_V = random.multivariate_normal(mean=[0, 0], cov=SIGMA, size=N)
            epsilon = eps_V[:, 0]
            V = eps_V[:, 1]
            X = Z @ PI + V
            Y = X * beta + epsilon
            
            PI_hat = inv(t(Z) @ Z) @ t(Z) @ X
            X_hat = Z @ PI_hat
            beta_hat = (1 / (t(X_hat) @ X_hat)) * t(X_hat) @ Y
            s_2 = (t(Y - X * beta_hat) @ (Y - X * beta_hat)) / (N - 1)
            var = s_2 * (1 / (t(X_hat) @ X_hat))
            tstat_result[i] = beta_hat / sqrt(var)
        rej_rate = mean((tstat_result > 1.96) | (tstat_result < -1.96))
        results[counter_a, counter_rho] = rej_rate
        print(rej_rate)
        counter_rho += 1
    counter_a += 1

print(results)
#%%


num_plots = results.shape[0]
colormap = plt.cm.gist_ncar
plt.gca().set_color_cycle([colormap(i)
                           for i in np.linspace(0, 0.9, num_plots)])

labels = []
for i in range(num_plots):
    plt.plot(rho_list, results[i, :])
    labels.append('a = ' + str(a_list[i]))

plt.legend(labels)
plt.xlabel('value of rho')
plt.ylabel('rejection frequency')
plt.show()


#%%
'''
EXERCISE 2
'''
k = 10
grid = [i * 0.25 for i in range(1000)]
S = 5000

results = zeros(len(grid))
counter = 0
for r_b0 in grid:
    random.seed(2110)
    LR = zeros(S)
    psi_1 = random.chisquare(1, size=S)
    psi_k = random.chisquare(k-1, size=S)
    LR = 1/2 * (psi_k + psi_1 - r_b0 + \
             sqrt(((psi_k + psi_1 + r_b0) ** 2) - 4 * r_b0 * psi_k))
    results[counter] = percentile(LR, q=95)
    counter += 1
print(results)

plt.plot(grid, results)
plt.xlabel('r(β)')
plt.ylabel('95% criticial value')
plt.show()


#%%
'''
EXERCISE 4
'''
k = 4
grid = [i * 0.25 for i in range(1000)]
S = 5000

results = zeros(len(grid))
counter = 0
for r_b0 in grid:
    random.seed(2110)
    LR = zeros(S)
    psi_1 = random.chisquare(1, size=S)
    psi_k = random.chisquare(k - 1, size=S)
    LR = 1 / 2 * (psi_k + psi_1 - r_b0 +
                  sqrt(((psi_k + psi_1 + r_b0) ** 2) - 4 * r_b0 * psi_k))
    results[counter] = percentile(LR, q=95)
    counter += 1
print(results)

plt.plot(grid, results)
plt.xlabel('r(β)')
plt.ylabel('95% criticial value')
plt.show()


#%%
'''
EXERCISE 3
'''


def sigma_eps_eps(Y, Z, X, beta_0):
    M_z = np.identity(n=N) - (Z @ inv(t(Z) @ Z) @ t(Z))
    return (1 / (N - k)) * t(Y - X * beta_0) @ M_z @ (Y - X * beta_0)
def sigma_eps_V(Y, Z, X, beta_0):
    M_z = np.identity(n=N) - Z @ inv(t(Z) @ Z) @ t(Z)
    sigma_eps_V = (1 / (N - k)) * t(Y - X * beta_0) @ M_z @ X
    return sigma_eps_V
def PI_tilde(Y, Z, X, beta_0):
    M_z = np.identity(n=N) - Z @ inv(t(Z) @ Z) @ t(Z) 
    sigma_e_e = sigma_eps_eps(Y, Z, X, beta_0)
    sigma_e_V = sigma_eps_V(Y, Z, X, beta_0)
    eps = Y - X * beta_0
    ro_hat = sigma_e_V / sigma_e_e
    PI_tilde = inv(t(Z) @ Z) @ t(Z) @ (X - eps * ro_hat)
    return(PI_tilde)


def AR(Y, Z, X, beta_0):
    P_z = Z @ inv(t(Z) @ Z) @ t(Z)
    return (t(Y - X * beta_0) @ P_z @ (Y - X * beta_0) / k) / \
           sigma_eps_eps(Y, Z, X, beta_0)


def LM(Y, Z, X, beta_0):
    Z_PI = Z @ PI_tilde(Y, Z, X, beta_0)
    P_zpi = np.outer(Z_PI, t(Z_PI)) * \
        ((t(Z_PI) @ Z_PI) ** (-1))
    LM = ((1 / sigma_eps_eps(Y, Z, X, beta_0)) *
          ((t(Y - X * beta_0)) @ P_zpi @ (Y - X * beta_0)))
    return LM


def r_beta_0(Y, Z, X, beta_0):
    M_z = np.identity(n=N) - Z @ inv(t(Z) @ Z) @ t(Z)
    SIGMA_VV = 1 / (N - k) * t(X) @ M_z @ X
    SIGMA_VV_eps = SIGMA_VV - \
                    ((sigma_eps_V(Y, Z, X, beta_0) ** 2) / sigma_eps_eps(Y, Z, X, beta_0))
    PI_tld = PI_tilde(Y, Z, X, beta_0)
    Z_PI = Z @ PI_tld
    r_beta = (1 / SIGMA_VV_eps) * \
        t(Y - X * beta_0) * (t(Z_PI) @ Z_PI) @ (Y - X * beta_0)
    return r_beta

def LR(Y, Z, X, beta_0):
    kAR = k * AR(Y=Y, Z=Z, X=X, beta_0=beta_0)
    r_b_0 = r_beta_0(Y=Y, Z=Z, X=X, beta_0=beta_0)
    LMstat = LM(Y=Y, Z=Z, X=X, beta_0=beta_0)
    LR = (kAR - r_b_0 + (sqrt((kAR + r_b_0) ** 2 - 4 * r_b_0 * (kAR - LMstat)))) / 2
    return LR


#%%
result_AR = zeros((len(a_list), len(rho_list)))
result_LM = zeros((len(a_list), len(rho_list)))
result_LR = zeros((len(a_list), len(rho_list)))
counter_a = 0
for a in a_list:
    PI = zeros(shape=k)
    PI[0] = a
    counter_rho = 0
    for rho in rho_list:
        SIGMA = np.array([[1, rho], [rho, 1]])
        AR_result_temp = zeros(REP)
        LM_result_temp = zeros(REP)
        LR_result_temp = zeros(REP)
        random.seed(2110)
        psi_k = np.random.chisquare(k - 1, size=REP)
        psi_1 = np.random.chisquare(1, size=REP)
        for i in range(REP):
            Z = random.normal(size=(N, k))
            eps_V = random.multivariate_normal(mean=[0, 0], cov=SIGMA, size=N)
            epsilon = eps_V[:, 0]
            V = eps_V[:, 1]
            X = Z @ PI + V
            Y = X * beta + epsilon

            # PI_hat = inv(t(Z) @ Z) @ t(Z) @ X
            # X_hat = Z @ PI_hat
            # beta_hat = (1 / (t(X_hat) @ X_hat)) * t(X_hat) @ Y
            # s_2 = (t(Y - X * beta_hat) @ (Y - X * beta_hat)) / (N - 1)
            # var = s_2 * (1 / (t(X_hat) @ X_hat))
            # AR_result_temp[i] = AR(Y=Y, Z=Z, X=X, beta_0=0)
            # LM_result_temp[i] = LM(Y=Y, Z=Z, X=X, beta_0=0)


            r_b0 = r_beta_0(Y=Y, Z=Z, X=X, beta_0=0)
            lr_b0 = (psi_k + psi_1 - r_b0 + np.sqrt((psi_k + psi_1 + r_b0) ** 2 - 4 * r_b0 * psi_k)) / 2
            LR_result_temp[i] = LR(Y=Y, Z=Z, X=X, beta_0=0) > np.percentile(lr_b0, q=95)


        rej_rate_AR = mean(AR_result_temp > 1.83)
        result_AR[counter_a, counter_rho] = rej_rate_AR
        rej_rate_LM = mean(LM_result_temp > 3.84)
        result_LM[counter_a, counter_rho] = rej_rate_LM
        rej_rate_LR = mean(LR_result_temp)
        result_LR[counter_a, counter_rho] = rej_rate_LR
        counter_rho += 1
    counter_a += 1
    print(counter_a)

# print(result_AR)
#%%
print(result_LR)
#%%
Z = random.normal(size=(N, k))
eps_V = random.multivariate_normal(mean=[0, 0], cov=SIGMA, size=N)
epsilon = eps_V[:, 0]
V = eps_V[:, 1]
X = Z @ PI + V
Y = X * beta + epsilon



# PI_hat = inv(t(Z) @ Z) @ t(Z) @ X
# X_hat = Z @ PI_hat
# beta_hat = (1 / (t(X_hat) @ X_hat)) * t(X_hat) @ Y
# s_2 = (t(Y - X * beta_hat) @ (Y - X * beta_hat)) / (N - 1)
# var = s_2 * (1 / (t(X_hat) @ X_hat))
print(LR(Y=Y, Z=Z, X=X, beta_0=0))
