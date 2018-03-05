# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 10:55:05 2018

@author: StepanAsus
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS as lm
from numpy import transpose as t
from numpy.linalg import inv as inv
from scipy.stats import chi2 as chi2

#%%

# Exercise 1
np.random.seed(1000)

mc_rep = 5000
n = 100
k = 10
a = np.array([1, 0.6, 0.3, 0.15, 0.07, 0.04, 0.02, 0])
rho = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.95])
e10 = np.zeros(k)
Z = np.random.normal(0, 1, ([n,k]))
t_stat = np.zeros([mc_rep])
rej_freq = np.zeros([len(a), len(rho)])

for i in range(len(a)):
  for p in range(len(rho)):
    np.random.seed(1000)
    for j in range(mc_rep):
      res = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1, rho[p]], [rho[p], 1]]), 
                                          size=(n))
      eps, v = res[:, 0], res[:, 1]
      pi = e10
      pi[0] = a[i]
      x = Z @ pi + v
      y = eps
      beta_2sls = (x.T @ Z @ inv(Z.T @ Z) @ Z.T @ x) ** (-1) * (x.T @ 
                        Z @ inv(Z.T @ Z) @ Z.T @ y)
      s_2sls = np.dot((y - x * beta_2sls).T, (y - x * beta_2sls)) / (n - 1)
      var_2sls = s_2sls * (x.T @ Z @ inv(Z.T @ Z) @ Z.T @ x) ** (-1)
      t_stat[j] = abs(beta_2sls / np.sqrt(var_2sls)) > 1.96
    rej_freq[i, t] = (np.mean(t_stat))
      

#%%
    
# Exercise 2
np.random.seed(1000)

r_b0 = np.arange(0, 200, step=0.5)
crit_values_lr = np.zeros([len(r_b0), ])

for i in range(len(r_b0)):
  psi_k = np.random.chisquare(k - 1, size=mc_rep)
  psi_1 = np.random.chisquare(1, size=mc_rep)
  
  lr_b0 = (psi_k + psi_1 - r_b0[i] + np.sqrt((psi_k + psi_1 + r_b0[i]) ** 2 - 4 * 
           r_b0[i] * psi_k)) / 2
  crit_values_lr[i] = np.percentile(lr_b0, q=95)
  
  
#%%
  
# Exercise 3
np.random.seed(1000)

mc_rep = 5000

# all in one loop
ar_stat = np.zeros([mc_rep])
rej_freq_ar = np.zeros([len(a), len(rho)])

lr_stat = np.zeros([mc_rep])
rej_freq_lr = np.zeros([len(a), len(rho)])

lm_stat = np.zeros([mc_rep])
rej_freq_lm = np.zeros([len(a), len(rho)])

for i in range(len(a)):
  for p in range(len(rho)):
    np.random.seed(1000)
    for j in range(mc_rep):
      # data
      res = np.random.multivariate_normal(mean=[0, 0], cov=np.array([[1, rho[p]], 
                                                                    [rho[p], 1]]), size=(n))
      eps, v = res[:, 0], res[:, 1]
      pi = e10
      pi[0] = a[i]
      x = Z @ pi + v
      y = eps
      pz = Z @ inv(Z.T @ Z) @ Z.T
      
      # AR
      ar_stat[j] = (y.T @ pz @ y) / (y.T @ (np.identity(pz.shape[0]) - pz) @ 
             y) * ((n - k) / k) 
      
      # LM
      rho_hat = (y.T @ (np.identity(pz.shape[0]) - pz) @ x) / (y.T @ 
           (np.identity(pz.shape[0]) - pz) @ y)
      pi_b0 = inv(Z.T @ Z) @ Z.T @ (x - y * rho_hat)
      pz_pi = np.outer(Z @ pi_b0, pi_b0.T @ Z.T) * ((pi_b0.T @ Z.T @ Z @ pi_b0) ** (-1))
      lm_stat[j] = (y.T @ pz_pi @ y) / (y.T @ (np.identity(pz.shape[0]) - pz) @ y) * (n - k)
      
      # LR
      psi_k = np.random.chisquare(k - 1, size=mc_rep)
      psi_1 = np.random.chisquare(1, size=mc_rep)
      r_b0 = (pi_b0.T @ Z.T @ Z @ pi_b0) / (((x.T @ 
             (np.identity(pz.shape[0]) - pz) @ x) - (((y.T @ 
             (np.identity(pz.shape[0]) - pz) @ x) ** 2) / 
             (y.T @ (np.identity(pz.shape[0]) - pz) @ y))) / (n - k))
      
      lr_b0 = (psi_k + psi_1 - r_b0 + np.sqrt((psi_k + psi_1 + r_b0) ** 2 - 4 * 
               r_b0 * psi_k)) / 2
      crit_values = np.percentile(lr_b0, q=95)
      lr_stat[j] = (1 / 2) * (k * ar_stat[j] - r_b0 + np.sqrt((k * ar_stat[j] + r_b0) ** (2) - 
             4 * r_b0 * (k * ar_stat[j] - lm_stat[j]))) > crit_values
    
    rej_freq_ar[i, p] = np.mean(ar_stat > (chi2.ppf(0.95, 10) / k))
    rej_freq_lr[i, p] = np.mean(lr_stat)
    rej_freq_lm[i, p] = np.mean(lm_stat > chi2.ppf(0.95, 1))


#%%
    
# Exercise 4
np.random.seed(1000)

r_b0 = np.arange(0, 200, step=0.5)
crit_values_lr_2 = np.zeros([len(r_b0), ])

for i in range(len(r_b0)):
  psi_k = np.random.chisquare(4, size=mc_rep)
  psi_1 = np.random.chisquare(1, size=mc_rep)
  
  lr_b0 = (psi_k + psi_1 - r_b0[i] + np.sqrt((psi_k + psi_1 + r_b0[i]) ** 2 - 4 * 
           r_b0[i] * psi_k)) / 2
  crit_values_lr_2[i] = np.percentile(lr_b0, q=95)


#%%
  
# data load for Exercise 5
np.random.seed(1000)

data = pd.read_csv("dest.csv", header=None)
data.columns = ["age", "age2", "ed", "exper", "exper2", "nearc2", "nearc4", "nearc4a", "nearc4b",
                      "race", "smsa", "South", "wage"]
y = data.loc[:, 'wage']
x = data.loc[:, 'ed']
z = data.loc[:, ("nearc2", "nearc4", "nearc4a", "nearc4b")]
w = np.hstack((data.loc[:, ("exper", "exper2", "race", "smsa", "South")].values, 
             np.ones(data.shape[0]).reshape((data.shape[0], 1))))

#%%

# Ex 5a, 2SLS and AR
np.random.seed(1000)

# t-stat confidence set
mw = np.identity(w.shape[0]) - np.array(w) @ inv(np.array(w).T @ np.array(w)) @ np.array(w).T
new_x = mw @ np.array(x)
new_y = mw @ np.array(y)
new_z = mw @ np.array(z)

# t-statistic
pz_5a = np.outer(new_z[:, 0], new_z[:, 0].T) * ((new_z[:, 0].T @ new_z[:, 0]) ** (- 1))
beta_2sls = ((new_x.T @ pz_5a @ new_x) ** (- 1)) * new_x.T @ pz_5a @ new_y
var_2sls = ((new_y - new_x * beta_2sls).T @ (new_y - new_x * beta_2sls) * ((new_x.T
           @ pz_5a @ new_x) ** (- 1))) / (3010 - 1)
se_2sls = np.sqrt(var_2sls)
t_stat_5a = beta_2sls / se_2sls
ci_2sls_5a = (beta_2sls - 1.96 * se_2sls, beta_2sls + 1.96 * se_2sls)

# AR onfidence set
betas = np.arange(-2, 2, 0.1)
ar_5a = np.zeros([len(betas), ])

for i in range(len(betas)):
  error = new_y - new_x * betas[i]
  ar_5a[i] = (error.T @ pz_5a @ error) / (error.T @ (np.identity(pz_5a.shape[0]) - pz_5a)
  @ error) * ((pz_5a.shape[0] - 1) / 1)

np.mean(ar_5a > (chi2.ppf(0.95, 1) / 1))
outcome_5a = pd.DataFrame([betas, ar_5a > (chi2.ppf(0.95, 1) / 1)]).T



#%%

# Exercise 5c
# first stage F-statistic

f_stat = (new_x.T @ pz_5a @ new_x) / (new_x.T @ (np.identity(pz_5a.shape[0]) - pz_5a) @ 
          new_x) * ((pz_5a.shape[0] - 1) / 1)


betas_5c = np.arange(10, 20, 0.25)
ar_5c = np.zeros([len(betas_5c), ])

for i in range(len(betas_5c)):
  error = new_y - new_x * betas_5c[i]
  ar_5c[i] = (error.T @ pz_5a @ error) / (error.T @ (np.identity(pz_5a.shape[0]) - pz_5a)
  @ error) * ((pz_5a.shape[0] - 1) / 1)



#%%

# Exercise 5e

np.random.seed(1000)

# t-statistic
pz_5e = new_z @ inv(new_z.T @ new_z) @ new_z.T
beta_2sls_5e = ((new_x.T @ pz_5e @ new_x) ** (- 1)) * new_x.T @ pz_5e @ new_y
var_2sls_5e = ((new_y - new_x * beta_2sls_5e).T @ (new_y - new_x * beta_2sls_5e) * ((new_x.T
           @ pz_5e @ new_x) ** (- 1))) / (3010 - 1)
se_2sls_5e = np.sqrt(var_2sls_5e)
t_stat_5e = beta_2sls_5e / se_2sls_5e
ci_2sls_5e = (beta_2sls_5e - 1.96 * se_2sls_5e, beta_2sls_5e + 1.96 * se_2sls_5e)

# AR, LM, LR confidence set
betas_5e = np.round(np.arange(-1, 2, 0.1), decimals=2)
ar_5e = np.zeros([len(betas_5e), ])
lm_5e = np.zeros([len(betas_5e), ])
lr_5e = np.zeros([len(betas_5e), ])
lr_5e_outcome = np.zeros([len(betas_5e), ])

for i in range(len(betas_5e)):
  error = new_y - new_x * betas_5e[i]
  
  # AR confidence set
  ar_5e[i] = (error.T @ pz_5e @ error) / (error.T @ (np.identity(pz_5e.shape[0]) - pz_5e)
  @ error) * ((pz_5e.shape[0] - 1) / 4)
  
  # LM confidence set
  rho_hat = (error.T @ (np.identity(pz_5e.shape[0]) - pz_5e) @ new_x) / (error.T @ 
     (np.identity(pz_5e.shape[0]) - pz_5e) @ error)
  pi_b0 = (inv(new_z.T @ new_z) @ new_z.T @ (new_x - error * rho_hat)).reshape(4, 1)
  pz_pi = (new_z @ pi_b0 @ pi_b0.T @ new_z.T) * ((pi_b0.T @ new_z.T @ new_z @ pi_b0) ** (-1))
  lm_5e[i] = (error.T @ pz_pi @ error) / (error.T @ (np.identity(pz_5e.shape[0]) - pz_5e) @ 
       error) * (pz_5e.shape[0] - 4)
  
  # LR confidence set
  psi_k = np.random.chisquare(3, size=mc_rep)
  psi_1 = np.random.chisquare(1, size=mc_rep)
  r_b0 = (pi_b0.T @ new_z.T @ new_z @ pi_b0) / (((new_x.T @ 
         (np.identity(pz_5e.shape[0]) - pz_5e) @ new_x) - (((error.T @ 
         (np.identity(pz_5e.shape[0]) - pz_5e) @ new_x) ** 2) / 
         (error.T @ (np.identity(pz_5e.shape[0]) - pz_5e) @ error))) / (pz_5e.shape[0] - 4))
  
  lr_b0 = (psi_k + psi_1 - r_b0 + np.sqrt((psi_k + psi_1 + r_b0) ** 2 - 4 * 
           r_b0 * psi_k)) / 2
  crit_value = np.percentile(lr_b0, q=95)
  lr_5e[i] = (1 / 2) * (4 * ar_5e[i] - r_b0 + np.sqrt((4 * ar_5e[i] + r_b0) ** (2) - 
         4 * r_b0 * (4 * ar_5e[i] - lm_5e[i])))
  lr_5e_outcome[i] = lr_5e[i] > crit_value


outcome_5e = pd.DataFrame([betas_5e, ar_5e, ar_5e > (chi2.ppf(0.95, 4) / 4), lm_5e,
                           lm_5e > chi2.ppf(0.95, 1), lr_5e, lr_5e_outcome]).T


#%%
  
np.min(ar_5e)













