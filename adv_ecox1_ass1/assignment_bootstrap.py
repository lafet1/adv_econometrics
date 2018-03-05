# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 20:23:14 2018

@author: StepanAsus
"""

import numpy as np


#%%
# instructions
np.random.seed(1429)
n = 50
rep = 1000
boot = 499
m = 0
s = 1.3
mu = np.exp(m + 1 / 2 * s**2)
beta = np.sin(mu)

xbar = np.zeros((rep, 1)) # average of original sample
bhat = np.zeros((rep, 1)) # estiamte of beta
se = np.zeros((rep, 1)) # standard error bhat (asymptotic)
trat = np.zeros((rep, 1)) # t-ratio
lcl_asym = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym = np.zeros((rep, 1)) # upper confidence limit (asym)


#%%
# for loop from instructions + question 1
np.random.seed(1429)

for i in range(rep):
  if i % 100 == 0:
    print(i)
  x = np.exp(np.random.normal(m, s, (n, 1)))
  xbar[i] = np.mean(x)
  bhat[i] = np.sin(xbar[i])
  se[i] = np.sqrt(np.cos(xbar[i]) * np.var(x) / n * (np.cos(xbar[i])))
  trat[i] = (bhat[i] - beta) / se[i]
  lcl_asym[i] = bhat[i] - 1.96 * se[i]
  ucl_asym[i] = bhat[i] + 1.96 * se[i]

cov_freq_asym = np.mean(np.logical_and(beta > lcl_asym, beta < ucl_asym))
print(cov_freq_asym)



#%%
# question 2
np.random.seed(1429)

for n in range(50, 2500, 50):
  for i in range(0, rep):
    if i % 100 == 0:
      print(i)
    x = np.exp(np.random.normal(m, s, (n, 1)))
    xbar[i] = np.mean(x)
    bhat[i] = np.sin(xbar[i])
    se[i] = np.sqrt(np.cos(xbar[i]) * np.var(x) / n * (np.cos(xbar[i])))
    trat[i] = (bhat[i] - beta) / se[i]
    lcl_asym[i] = bhat[i] - 1.96 * se[i]
    ucl_asym[i] = bhat[i] + 1.96 * se[i]
  if 0.9365 < np.mean(np.logical_and(beta > lcl_asym, beta < ucl_asym)) < 0.9635:
    minimal = n
    break
  
print(minimal)
    

#%%
# question 3
np.random.seed(1429)
n = 50

# bootstrap/jackknife equivalents
xbar_b = np.zeros((rep, 1)) # average of original sample
bias_corr_b = np.zeros((rep, 1)) # estiamte of beta
bias_corr_jk = np.zeros((rep, 1)) # estiamte of beta
se_b = np.zeros((rep, 1)) # standard error bhat (asymptotic)
trat_b = np.zeros((rep, 1)) # t-ratio
trat_jk = np.zeros((rep, 1)) # t-ratio
lcl_asym_b = np.zeros((rep, 1)) # lower confidence limit (asym)
lcl_asym_jk = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym_b = np.zeros((rep, 1)) # upper confidence limit (asym)
ucl_asym_jk = np.zeros((rep, 1)) # upper confidence limit (asym)

for i in range(rep):
  # we generate the data and then get the standard estimate
  x = np.exp(np.random.normal(m, s, (n, 1)))

  # bootstrap
  xbar_aux = np.zeros((boot, 1))
  aux_bhat = np.zeros((boot, 1))
  for k in range(boot):
    index = np.random.randint(0, n - 1, n)
    xbar_aux[k] = np.mean(x[[index]])
    aux_bhat[k] = np.sin(xbar_aux[k])
  
  beta_b = np.mean(aux_bhat)
  bias_corr_b[i] = 2 * np.sin(np.mean(x)) - beta_b
  
  xbar_b[i] = np.mean(x)
  se_b[i] = np.sqrt(np.cos(xbar_b[i]) * np.var(x) / n * (np.cos(xbar_b[i])))
  trat_b[i] = (bias_corr_b[i] - beta) / se_b[i]
  lcl_asym_b[i] = bias_corr_b[i] - 1.96 * se_b[i]
  ucl_asym_b[i] = bias_corr_b[i] + 1.96 * se_b[i]
  
  # jackknife
  xbar_aux = np.zeros((n, 1))
  aux_bhat = np.zeros((n, 1))
  for k in range(n):
    ind = np.ones(n, bool)
    ind[k] = False
    xbar_aux[k] = np.mean(x[ind])
    aux_bhat[k] = np.sin(xbar_aux[k])
    
  beta_jk = np.mean(aux_bhat)
  bias_corr_jk[i] = n * np.sin(np.mean(x)) - (n - 1) * beta_jk

  trat_jk[i] = (bias_corr_jk[i] - beta) / se_b[i]
  lcl_asym_jk[i] = bias_corr_jk[i] - 1.96 * se_b[i]
  ucl_asym_jk[i] = bias_corr_jk[i] + 1.96 * se_b[i]


cov_freq_asym_b = np.mean(np.logical_and(beta > lcl_asym_b, beta < ucl_asym_b))
cov_freq_asym_jk = np.mean(np.logical_and(beta > lcl_asym_jk, beta < ucl_asym_jk))

print(cov_freq_asym_b)
print(cov_freq_asym_jk)



#%%
# question 4
np.random.seed(1429)
n = 50

# reseting the previous arrays
se_b = np.zeros((rep, 1)) # standard error bhat (asymptotic)
se_jk = np.zeros((rep, 1)) # standard error bhat (asymptotic)
trat_b2 = np.zeros((rep, 1)) # t-ratio
trat_jk2 = np.zeros((rep, 1)) # t-ratio
lcl_asym_b2 = np.zeros((rep, 1)) # lower confidence limit (asym)
lcl_asym_jk2 = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym_b2 = np.zeros((rep, 1)) # upper confidence limit (asym)
ucl_asym_jk2 = np.zeros((rep, 1)) # upper confidence limit (asym)


for i in range(rep):
  # generate data
  x = np.exp(np.random.normal(m, s, (n, 1)))
  xbar[i] = np.mean(x)
  bhat[i] = np.sin(xbar[i])

  # bootstrap
  xbar_aux = np.zeros((boot, 1))
  aux_bhat = np.zeros((boot, 1))
  for k in range(boot):
    index = np.random.randint(0, n - 1, n)
    xbar_aux[k] = np.mean(x[[index]])
    aux_bhat[k] = np.sin(xbar_aux[k])
  
  beta_b = np.mean(aux_bhat)
  
  se_b[i] = np.sqrt((1 / (boot - 1)) * np.sum((aux_bhat - beta_b) ** 2))
  trat_b2[i] = (bhat[i] - beta) / se_b[i]
  lcl_asym_b2[i] = bhat[i] - 1.96 * se_b[i]
  ucl_asym_b2[i] = bhat[i] + 1.96 * se_b[i]
  
  # jackknife
  xbar_aux = np.zeros((n, 1))
  aux_bhat = np.zeros((n, 1))
  for k in range(n):
    ind = np.ones(n, bool)
    ind[k] = False
    xbar_aux[k] = np.mean(x[ind])
    aux_bhat[k] = np.sin(xbar_aux[k])
    
  beta_jk = np.mean(aux_bhat)

  se_jk[i] = np.sqrt(((n - 1) / n) * np.sum((aux_bhat - beta_jk) ** 2))
  trat_jk2[i] = (bhat[i] - beta) / se_jk[i]
  lcl_asym_jk2[i] = bhat[i] - 1.96 * se_jk[i]
  ucl_asym_jk2[i] = bhat[i] + 1.96 * se_jk[i]


cov_freq_asym_b2 = np.mean(np.logical_and(beta > lcl_asym_b2, beta < ucl_asym_b2))
cov_freq_asym_jk2 = np.mean(np.logical_and(beta > lcl_asym_jk2, beta < ucl_asym_jk2))

print(cov_freq_asym_b2)
print(cov_freq_asym_jk2)


#%%
# question 5
np.random.seed(1429)


perc_boot = np.zeros((rep, 2))
lcl_asym_q5 = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym_q5 = np.zeros((rep, 1)) # upper confidence limit (asym)

for i in range(rep):
  # generate data
  x = np.exp(np.random.normal(m, s, (n, 1)))
  xbar = np.mean(x)
  bhat = np.sin(xbar)
  se = np.sqrt(np.cos(xbar) * np.var(x) / n * (np.cos(xbar)))

  # bootstrap
  xbar_aux = np.zeros((boot, 1))
  aux_bhat = np.zeros((boot, 1))
  aux_trat = np.zeros((boot, 1))
  aux_se = np.zeros((boot, 1))
  for k in range(boot):
    index = np.random.randint(0, n - 1, n)
    xbar_aux[k] = np.mean(x[[index]])
    aux_bhat[k] = np.sin(xbar_aux[k])
    aux_se[k] = np.sqrt(np.cos(xbar_aux[k]) * np.var(x[[index]]) / n * (np.cos(xbar_aux[k])))
  
  # percentile
  perc_boot[i] = (np.percentile(aux_bhat, 2.5, interpolation='higher'), 
           np.percentile(aux_bhat, 97.5, interpolation='higher'))
  
  beta_b = np.mean(aux_bhat)
  
  # percentile-t
  for k in range(boot):
    aux_trat[k] = (aux_bhat[k] - bhat) / aux_se[k]
  ci_tprec_boot = (np.percentile(aux_trat, 2.5, interpolation='higher'), 
                  np.percentile(aux_trat, 97.5, interpolation='higher'))  
    
  lcl_asym_q5[i] = bhat - ci_tprec_boot[1] * se
  ucl_asym_q5[i] = bhat - ci_tprec_boot[0] * se
  
  
cov_freq_asym_q5 = (np.mean(np.logical_and(beta > lcl_asym_q5, beta < ucl_asym_q5)), 
                    np.mean(np.logical_and(beta > perc_boot[:, 0], beta < perc_boot[:, 1]), axis=tuple(range(0, 1))))

print(cov_freq_asym_q5)


#%%
# question 6
np.random.seed(1429)

lcl_asym_b3 = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym_b3 = np.zeros((rep, 1)) # upper confidence limit (asym)

for i in range(rep):
  # generate data
  x = np.exp(np.random.normal(m, s, (n, 1)))
  xbar = np.mean(x)
  bhat = np.sin(xbar)
  se = np.sqrt(np.cos(xbar) * np.var(x) / n * (np.cos(xbar)))
  
  # bootstrap
  xbar_aux = np.zeros((boot, 1))
  aux_bhat = np.zeros((boot, 1))
  aux_trat = np.zeros((boot, 1))
  aux_se = np.zeros((boot, 1))
  for k in range(boot):
    index = np.random.randint(0, n - 1, n)
    xbar_aux[k] = np.mean(x[[index]])
    aux_bhat[k] = np.sin(xbar_aux[k])
    aux_se[k] = np.sqrt(np.cos(xbar_aux[k]) * np.var(x[[index]]) / n * (np.cos(xbar_aux[k])))
  
  beta_b = np.mean(aux_bhat)
  bias_corr_b = 2 * np.sin(np.mean(x)) - beta_b
  
  # percentile-t
  for k in range(boot):
    aux_trat[k] = (aux_bhat[k] - bhat) / aux_se[k]
  ci_tprec_boot = (np.percentile(aux_trat, 2.5, interpolation='higher'), 
                  np.percentile(aux_trat, 97.5, interpolation='higher'))  
    
  lcl_asym_b3[i] = bias_corr_b - ci_tprec_boot[1] * se
  ucl_asym_b3[i] = bias_corr_b - ci_tprec_boot[0] * se

cov_freq_asym_b3 = np.mean(np.logical_and(beta > lcl_asym_b3, beta < ucl_asym_b3))

print(cov_freq_asym_b3)


#%%
# question 7
np.random.seed(1429)

lcl_asym_4 = np.zeros((rep, 1)) # lower confidence limit (asym)
ucl_asym_4 = np.zeros((rep, 1)) # upper confidence limit (asym)

for i in range(rep):
  # generate data
  x = np.exp(np.random.normal(m, s, (n, 1)))
  xbar = np.mean(x)
  bhat = np.sin(xbar)
  se = np.sqrt(np.cos(xbar) * np.var(x) / n * (np.cos(xbar)))
  
  # jackknife - bias correction
  xbar_aux = np.zeros((n, 1))
  aux_bhat = np.zeros((n, 1))
  for k in range(n):
    mask = np.ones(n, dtype=bool)
    mask[k] = 0
    xbar_aux[k] = np.mean(x[[mask]])
    aux_bhat[k] = np.sin(xbar_aux[k])
  
  beta_jk = np.mean(aux_bhat)
  bias_jk = (n - 1) * (beta_jk - bhat)
  
  # bootstrap
  xbar_aux = np.zeros((boot, 1))
  aux_bhat = np.zeros((boot, 1))
  aux_trat = np.zeros((boot, 1))
  aux_se = np.zeros((boot, 1))
  for k in range(boot):
    index = np.random.randint(0, n - 1, n)
    xbar_aux[k] = np.mean(x[[index]])
    aux_bhat[k] = np.sin(xbar_aux[k])
    aux_se[k] = np.sqrt(np.cos(xbar_aux[k]) * np.var(x[[index]]) / n * (np.cos(xbar_aux[k])))
    aux_trat[k] = (aux_bhat[k] - bias_jk - bhat) / aux_se[k]
  
  tprec_boot_bc4 = (np.percentile(aux_trat, 2.5, interpolation='higher'), 
                    np.percentile(aux_trat, 97.5, interpolation='higher'))
  
  lcl_asym_4[i] = bhat - tprec_boot_bc4[1] * se
  ucl_asym_4[i] = bhat - tprec_boot_bc4[0] * se

cov_freq_asym_4 = np.mean(np.logical_and(beta > lcl_asym_4, beta < ucl_asym_4))

print(cov_freq_asym_4)







  
  
  
  
