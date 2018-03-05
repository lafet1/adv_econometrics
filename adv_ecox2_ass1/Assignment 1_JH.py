# %  Name                   	Student id                email
# % +------------------------+------------------------+-------------------------
# % |       Jan Hynek        |      11748494          | janhynek@gmail.com
# % +------------------------+------------------------+-------------------------
# % |     Stepan Svoboda     |      11762616          | stepan.svo@gmail.com
# % +------------------------+------------------------+-------------------------
# % I (enlisted above) declare that:
# %   1. Our assignment will be our own work.
# %   2. We shall not make solutions to the assignment available to anyone else.
# %   3. We shall not engage in any other activities that will dishonestly improve my results or dishonestly improve or hurt the results of others.

#%%

######################
##### DEFINITION #####
######################


# For easier manipulation with functions we decided to import several functions from numpy package.
import numpy as np
from numpy import sin, cos, exp, mean, sqrt, percentile, std, var, zeros
np.random.seed(2110)
#%%
n = 50
print("n: " + str(n))
REP = 1000

print("MC rep: " +str(REP))
BOOTREP = 499
print('Bootstrap rep: ' + str(BOOTREP))
m = 0
s = 1.3

bhat = exp(m + (1/2 * s ** 2))
print('bhat: ' + str(bhat))
beta = sin(bhat)
print('beta: ' + str(beta))

# % average original sample
xbar = zeros(REP)
# % estimate of beta
bhat = zeros(REP)
# % estimate of beta
bhat_jk = zeros(REP)
# % estimate of beta
bhat_bs = zeros(REP)
# % standard error bhat(asymptotic)
SE = zeros(REP)
# Bootstrap estimate of standard errors
SE_bs = zeros(REP)
# jackknife estimate of SE
SE_jk = zeros(REP)
# % t - ratio
trat = zeros(REP)
# % Lower confidence limit(asym)
LCLasym = zeros(REP)
# % Upper confidence limit(asym)
UCLasym = zeros(REP)
# % Lower confidence limit(JK)
LCL_jk = zeros(REP)
# % Upper confidence limit(jackknife)
UCL_jk = zeros(REP)
# % Lower confidence limit(bootstrap)
LCL_bs = zeros(REP)
# % Upper confidence limit(bootstrap)
UCL_bs = zeros(REP)
# Percentiles, used in ex 5,6,7
lower_perc = zeros(REP)
upper_perc = zeros(REP)
lower_tstat = zeros(REP)
upper_tstat = zeros(REP)

#%%
'''
######################
##### EXERCISE 1 #####
######################
'''

# This exercise provides us with benchmark. 
# We would like to have better results than ordinary montecarlo without bootstrap
# Other than that, we calculated standard errors using delta method
n = 50
for i in range(REP):
    if i%100 == 0: 
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size = n))
    xbar[i] = mean(X)
    bhat[i] = sin(xbar[i])
    SE[i] = sqrt(var(X) * 1/n) * np.abs(cos(xbar[i]))
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - 1.96 * SE[i]
    UCLasym[i] = bhat[i] + 1.96 * SE[i]
print('\n \n Number of observations: ' + str(n))
CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
print('Coverage freq.(asym):'   + str(CoverageFreqasym))


#%%
'''
######################
##### EXERCISE 2 #####
######################
'''
# In this exercise we run a while loop(In which we run for loop, from previous question) 
# and we add more observations, until we converge into specified confidence interval. 
# This is usually between 1.5k - 2k observations. Depends on seed.
CoverageFreqasym = 0
n = 50
report_numbers = False
while (CoverageFreqasym < 0.9365) or (CoverageFreqasym > 0.9635):
    for i in range(REP):
        if (i % 100) == 0 and report_numbers:
            print(i)
        X = exp(np.random.normal(loc=m, scale=s, size=n))
        xbar[i] = mean(X)
        bhat[i] = sin(xbar[i])
        SE[i] = sqrt(var(X) * 1 / n) * np.abs(cos(xbar[i]))
        trat[i] = (bhat[i] - beta) / SE[i]
        LCLasym[i] = bhat[i] - 1.96 * SE[i]
        UCLasym[i] = bhat[i] + 1.96 * SE[i]
    CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
    n += 50
print('n, for which coverage frequency is [0.9365, 0.9635]: '+ str(n - 50))


#%%
'''
###############################
##### Function definition #####
###############################
'''
# In this part, I define functions which will be used in next exercises as well.


def calculate_orig(X):
    '''
    Take list of observations X 
    and calculate mean, SE using delta method and bhat (original theta)
    '''
    xbar = mean(X)
    bhat = sin(xbar)
    SE = sqrt(var(X) * 1 / n) * np.abs(cos(xbar))
    return xbar, bhat, SE

def calculate_jk(X):
    '''
    Take list of observations X and
    a) using a for loop, calculate jackknife estimate.
    b) return bias corrected jackknife estimate, standard error, 
    and bias correction (useful for Ex7)
    '''
    N = len(X)
    bhat_jk = zeros(N)
    for i in range(N):
        jk_X = X[:i] + X[(i + 1):]
        bhat_X_jk = mean(jk_X)
        bhat_jk[i] =sin(bhat_X_jk)
    theta_jk = mean(bhat_jk)
    theta_jk_BC = N * sin(mean(X)) - (N - 1) * theta_jk
    bias_jk = (N-1) * (theta_jk - sin(mean(X)))
    se_jk = sqrt((N - 1) * var(bhat_jk))
    return theta_jk_BC, se_jk, bias_jk


def calculate_bs(X, bootrep):
    '''
    Take a list of observation X, along with number of repetitions bootrep and:
    calculate bootstrap estimate and return it, along with calculated standard error.
    '''
    N = len(X)
    bhat_bs = zeros(bootrep)
    for i in range(bootrep):
        indices = np.random.randint(0, high = N, size = N)
        bootstrap = [X[j] for j in indices.tolist()]
        bhat_bs[i] = sin(mean(bootstrap))
    theta_bs = 2 * sin(mean(X)) - mean(bhat_bs)
    se_bs = std(bhat_bs)
    return theta_bs, se_bs

#%%
'''
######################
##### EXERCISE 3 #####
######################
'''

# In this section we use jackknfe and bootstrap to calculate coverage frequency
# We do a montecarlo simulation, and this is done using already specified functions
# In this exercise we change, in contrast with Ex1, the bhat estimates only
# We can observe that these estimates of CI are not better than original estimate.
# orig: 0.74
# BS: 0.695
# JK: 0.691


n = 50
report_numbers = True
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i], bias = calculate_jk(X.tolist())
    bhat_bs[i], SE_bs[i] = calculate_bs(X.tolist(), BOOTREP)
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - 1.96 * SE[i]
    UCLasym[i] = bhat[i] + 1.96 * SE[i]
    LCL_jk[i] = bhat_jk[i] - 1.96 * SE[i]
    UCL_jk[i] = bhat_jk[i] + 1.96 * SE[i]
    LCL_bs[i] = bhat_bs[i] - 1.96 * SE[i]
    UCL_bs[i] = bhat_bs[i] + 1.96 * SE[i]
CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
CoverageFreq_jk = mean((beta > LCL_jk) & (beta < UCL_jk))
CoverageFreq_bs = mean((beta > LCL_bs) & (beta < UCL_bs))
print('\n \n Coverage frequency using')
print('asymptotic estimates: ' + str(CoverageFreqasym))
print('BS estimates: ' + str(CoverageFreq_bs))
print('JK estimates: ' + str(CoverageFreq_jk))

#%%
'''
######################
##### EXERCISE 4 #####
######################
'''

# This exercise is similar to previous exercise,
# however this time we are changing the standard error estimates only, 
#  obtained from bootstrap and jackknife.


n = 50
report_numbers = True
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i], bias = calculate_jk(X.tolist())
    bhat_bs[i], SE_bs[i] = calculate_bs(X.tolist(), BOOTREP)
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - 1.96 * SE[i]
    UCLasym[i] = bhat[i] + 1.96 * SE[i]
    LCL_jk[i] = bhat[i] - 1.96 * SE_jk[i]
    UCL_jk[i] = bhat[i] + 1.96 * SE_jk[i]
    LCL_bs[i] = bhat[i] - 1.96 * SE_bs[i]
    UCL_bs[i] = bhat[i] + 1.96 * SE_bs[i]
CoverageFreqasym = mean((beta > LCLasym) & (beta < UCLasym))
cov_freq_jk = mean((beta > LCL_jk) & (beta < UCL_jk))
cov_freq_bs = mean((beta > LCL_bs) & (beta < UCL_bs))
print('\n \n Coverage frequency using')
print('asymptotic SE\'s: ' + str(CoverageFreqasym))
print('BS SE\'s: ' + str(cov_freq_bs))
print('JK SE\'s: ' + str(cov_freq_jk))
#%%

'''
######################
##### EXERCISE 5 #####
######################
'''

# In this exercise we implement method in which we estimate confidence intervals using
#  a) percentile method
#  b) percentile-t method
# However, these methods are very computationally expensive.

def calculate_bs_percentiles(X, bootrep):
    '''
    Main reason for this function is to obtain confidence interval estimates using
  a) percentile method - obtaining all bootstrap estimates, 
     and afterwards obtaining confidence interval from these obtained estimates
     by obtaining value of 2.5 percentile and 97.5 percentile of the bootstrapped thetas.
  b) percentile-t method - in every bootstrap, we calculate t-statistic.
     SE for this t-stat is obtained using delta method from the bootstrapped sample.
    '''
    N = len(X)
    bhat_bs = zeros(bootrep)
    tstat = zeros(bootrep)
    bhat = sin(mean(X))
    for i in range(bootrep):
        indices = np.random.randint(0, high=(N-1), size=N)
        bootstrap = [X[j] for j in indices.tolist()]
        bhat_bs[i] = sin(mean(bootstrap))
        
        SE = sqrt(var(bootstrap) * 1 / n) * np.abs(cos(mean(bootstrap)))
        tstat[i] = (bhat_bs[i] - bhat) / SE

    se_bs = sqrt(sum([(i - mean(bhat_bs)) ** 2 for i in bhat_bs]))/bootrep
    # tstat = [(i - bhat) / se_bs for i in bhat_bs.tolist()]
    theta_bs = 2 * bhat - mean(bhat_bs)
    lower_perc = np.percentile(bhat_bs, 2.5)
    upper_perc = np.percentile(bhat_bs, 97.5)
    lower_tstat = np.percentile(tstat, 2.5)
    upper_tstat = np.percentile(tstat, 97.5)
    return theta_bs, se_bs, lower_perc, upper_perc, lower_tstat, upper_tstat



# Monte Carlo simulation
# Now we use specified function to calculate  the bootstrapped values.
n = 50
report_numbers = True
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i], bias = calculate_jk(X.tolist())
    (bhat_bs[i], SE_bs[i],
     lower_perc[i], upper_perc[i],
     lower_tstat[i], upper_tstat[i]) = calculate_bs_percentiles(X.tolist(), BOOTREP)
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = lower_perc[i]
    UCLasym[i] = upper_perc[i]
    LCL_jk[i] = bhat[i] - upper_tstat[i] * SE[i]
    UCL_jk[i] = bhat[i] - lower_tstat[i] * SE[i]

    

percentile_method = mean((beta > LCLasym) & (beta < UCLasym))
percentile_t_method = mean((beta > LCL_jk) & (beta < UCL_jk))
print('\n \n Coverage Frequency using:')
print('percentile: ' + str(percentile_method))
print('percentile-t: ' + str(percentile_t_method))


#%%

'''
######################
##### EXERCISE 6 #####
######################
'''
# In this exercise we compare the percentiles
# when using the centres of the intervals bias corrected estimates
# This is done by changing bhat for bhat estimated from both bootstrap and jackknife
# We can observe from the results no added benefit - the results are rather worse.

n = 50
report_numbers = True
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    bhat_jk[i], SE_jk[i], bias = calculate_jk(X.tolist())
    (bhat_bs[i], SE_bs[i],
     lower_perc[i], upper_perc[i],
     lower_tstat[i], upper_tstat[i]) = calculate_bs_percentiles(X.tolist(), BOOTREP)
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
    trat[i] = (bhat[i] - beta) / SE[i]
    LCL_jk[i] = bhat_jk[i] - upper_tstat[i] * SE[i]
    UCL_jk[i] = bhat_jk[i] - lower_tstat[i] * SE[i]
    LCL_bs[i] = bhat_bs[i] - upper_tstat[i] * SE[i]
    UCL_bs[i] = bhat_bs[i] - lower_tstat[i] * SE[i]
    LCLasym[i] = bhat[i] - upper_tstat[i] * SE[i]
    UCLasym[i] = bhat[i] - lower_tstat[i] * SE[i]
percentile_t = mean((beta > LCLasym) & (beta < UCLasym))
percentile_t_bs = mean((beta > LCL_bs) & (beta < UCL_bs))
percentile_t_jk = mean((beta > LCL_jk) & (beta < UCL_jk))
# cov_freq_bs = mean((beta > LCL_bs) & (beta < UCL_bs))

print('\n \n Coverage Frequency using:')
print('orig percentile-t: ' + str(percentile_t))
print('percentile-t JK: ' + str(percentile_t_jk))
# print('BS: ' + str(cov_freq_bs))
print('percentile-t BS: ' + str(percentile_t_bs))

#%%

'''
######################
##### EXERCISE 7 #####
######################
'''
#  In this exercise we calculated bias corrected t-statistic in every bootstrap iteration.
# Results from this method are the best of all calculated methods. However, this comes
# at a cost - calculating these results takes the longest time

def calculate_bs_with_jk(X, bootrep):
    '''
    Calculate bootstrap percentile-t CI using jackknife corrected t-statistc
    This function performs bootstrap on a sample, and in the bootstrap it calculates jackknife bias
    This bias is afterwards subtracted from original bootstrapped estimate (bias correction)
    Calculation of standard error of bootstrapped bhat is done using delta method done on bootstrapped sample
    Function returns all possible t statistics.

    '''
    N = len(X)
    bhat_bs = zeros(bootrep)
    tstat = zeros(bootrep)
    tstat_corr = zeros(bootrep)
    bhat = sin(mean(X))
    for i in range(bootrep):
        indices = np.random.randint(0, high=(N - 1), size=N)
        bootstrap = [X[j] for j in indices.tolist()]
        bhat_bs[i] = sin(mean(bootstrap))
        a, b,  bias_jk = calculate_jk(bootstrap)
        SE = sqrt(var(bootstrap) * 1 / n) * np.abs(cos(mean(bootstrap)))
        tstat[i] = (bhat_bs[i] - bhat) / SE
        tstat_corr[i] = (bhat_bs[i] - bias_jk - bhat) / SE

    se_bs = sqrt(sum([(i - mean(bhat_bs)) ** 2 for i in bhat_bs])) / bootrep
    # tstat = [(i - bhat) / se_bs for i in bhat_bs.tolist()]
    theta_bs = 2 * bhat - mean(bhat_bs)
    lower_perc = np.percentile(bhat_bs, 2.5)
    upper_perc = np.percentile(bhat_bs, 97.5)
    lower_tstat = np.percentile(tstat, 2.5)
    upper_tstat = np.percentile(tstat, 97.5)
    lower_tstat_corr = np.percentile(tstat_corr, 2.5)
    upper_tstat_corr = np.percentile(tstat_corr, 97.5)
    return theta_bs, se_bs, \
           lower_perc, upper_perc, \
           lower_tstat, upper_tstat, \
           lower_tstat_corr, upper_tstat_corr


bhat_bs_corr = zeros(REP)
SE_bs_corr = zeros(REP)
lower_perc_corr = zeros(REP)
upper_perc_corr = zeros(REP)
lower_tstat_corr = zeros(REP)
upper_tstat_corr = zeros(REP)
LCL_corr = zeros(REP)
UCL_corr = zeros(REP)
n = 50
report_numbers = True
for i in range(REP):
    if (i % 100) == 0 and report_numbers:
        print(i)
    X = exp(np.random.normal(loc=m, scale=s, size=n))
    xbar[i], bhat[i], SE[i] = calculate_orig(X.tolist())
    bhat_jk[i], SE_jk[i], bias = calculate_jk(X.tolist())
    # (bhat_bs[i], SE_bs[i],
    #  lower_perc[i], upper_perc[i],
    #  lower_tstat[i], upper_tstat[i]) = calculate_bs_percentiles(X.tolist(), BOOTREP)
    (bhat_bs[i], SE_bs[i],
     lower_perc[i], upper_perc[i],
     lower_tstat[i], upper_tstat[i],
     lower_tstat_corr[i], upper_tstat_corr[i]) = calculate_bs_with_jk(X.tolist(), BOOTREP)

    trat[i] = (bhat[i] - beta) / SE[i]
    LCLasym[i] = bhat[i] - upper_tstat[i] * SE[i]
    UCLasym[i] = bhat[i] - lower_tstat[i] * SE[i]
    LCL_corr[i] = bhat[i] - upper_tstat_corr[i] * SE[i]
    UCL_corr[i] = bhat[i] - lower_tstat_corr[i] * SE[i]
percentile_t = mean((beta > LCLasym) & (beta < UCLasym))
percentile_t_corr = mean((beta > LCL_corr) & (beta < UCL_corr))
# cov_freq_bs = mean((beta > LCL_bs) & (beta < UCL_bs))

print('\n \n Coverage Frequency using:')
print('orig percentile-t: ' + str(percentile_t))
# print('BS: ' + str(cov_freq_bs))
print('percentile-t corrected: ' + str(percentile_t_corr))
