#%%

'''
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
'''

#%%

import numpy as np
from sklearn import linear_model
from numpy.linalg import *
from numpy import transpose as t
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd


#%%

studentname1 = 'Jan Hynek'
studentname2 = 'Štěpán Svoboda'
studentnumber1 = 11748494
studentnumber2 = 11762616
np.random.seed(studentnumber1 + 10 * studentnumber2)
print('This is the Python output for Assignment 1, Advanced Econometrics 1 ' +
      'for students: \n' +
      '   (1) ' + studentname1 + ' - ' + str(studentnumber1) + '\n' +
      '   (2) ' + studentname2 + ' - ' + str(studentnumber2) + '\n\n' +
      '   Group 7\n')


# global variables
# in this assignment, almost no variables are global.
beta = 1


def cov_matrix(sigma_sq_z, gamma, rho, r):
    '''
    sigma_sq_z = variance of matrix Z
    gamma = covariance between u an Z - instrument validity
    rho = covariance between u and X - level of endogeneity
    r = number of instruments


    This function takes all given arguments and returns covariance matrix.
    '''
    cov_matrix = np.identity(r + 2) * sigma_sq_z
    cov_matrix[:, 0] = (r + 2) * [gamma]
    cov_matrix[0, :] = (r + 2) * [gamma]
    cov_matrix[0, 0], cov_matrix[1, 1] = 1, 1
    cov_matrix[0, 1], cov_matrix[1, 0] = rho, rho
    return np.matrix(cov_matrix)


# defined as three values = [u, v, epsilon]


def errors(cov_matrix, N_obs, r):
    '''
    cov_matrix = covariance matrix
    N_obs = number of observations
    r = number of instruments

    wrapper for multivariate_normal generator. generates errors u, v, epsilon.
    returns matrix of all errors. 0th column is u, 1st is v,
    all other are epsilon.

    '''
    errors_vector = np.random.multivariate_normal(mean=[0] * (r + 2),
                                                  cov=cov_matrix,
                                                  size=N_obs)
    return np.matrix(errors_vector)


def iteration(gamma, rho, pi_1, N_obs, sigma_sq_z, r):
    '''
    sigma_sq_z = variance of matrix Z
    gamma = covariance between u an Z - instrument validity
    rho = covariance between u and X - level of endogeneity
    r = number of instruments
    N_obs = number of observations
    pi_1 = true coefficient between x and z - instrument relevance

    This function takes all arguments and forms errors
    Afterwards, it forms the 'true values' of Z, X and y from the specified
    models.
    Then, two regressions are computed.
    First, we calculate OLS using sklearn linear_model and obtain b_hat_ols
    estimator.
    Then, we use (X'X)^-1X'*omega_hat*X(X'X)^-1
    to calculate variance of b_hat_ols.
    This is probably not needed as we have homoskedasticity but we opted for the
    more general route.

    Second, we calculate 2SLS estimator in two regressions.
    In the first, we obtain estimator Pi_hat, which estimates the effect of
    Z on X.
    We use Pi_hat to estimate X_hat, which we afterwards use to estimate
    to estimate y.
    We obtain estimator b_hat_2sls, and in the end
    we calculate variance similarily.

    This function returns list of calculated betas and variances,
    along with concentration parameter, computated from true values.

    '''
    pi = t(np.matrix([pi_1] + [0] * (r - 1)))
    error_matrix = errors(cov_matrix(sigma_sq_z, gamma, rho, r), N_obs, r)
    # u = error_matrix[:, 0]
    # v = error_matrix[:, 1]

    Z = error_matrix[:, 2:]
    X = Z * pi + error_matrix[:, 1]
    y = X * beta + error_matrix[:, 0]

    #####################
    #####################
    ## estimating part ##
    #####################
    #####################

    ##########################
    # Ordinary least squares #
    ##########################
    # we do not fit intercept as we know
    # that our original model does not have intercept.
    # however, we think that in monte carlo simulation
    # it would not make a significant difference if the intercept would be zero
    # as it is estimated as almost zero
    regression_ols = linear_model.LinearRegression(fit_intercept=False)
    regression_ols.fit(X=X, y=y)
    b_hat_ols = regression_ols.coef_
    # b_hat_ols2 = inv(t(X) * X) * t(X) * y
    # print(b_hat_ols - b_hat_ols2)
    u_hat_ols = y - X * b_hat_ols
    Omega_hat_ols = np.diag(np.power(u_hat_ols.A1, 2))
    Var_b_ols = (inv(t(X) * X) * t(X) *
                 Omega_hat_ols *
                 X * inv(t(X) * X))

    ###########################
    # Two stage least squares #
    ###########################

    # first regression
    regression_2sls_1 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_1.fit(X=Z, y=X)
    Pi_hat = t(np.matrix(regression_2sls_1.coef_))
    X_hat = Z * Pi_hat

    # second regression
    regression_2sls_2 = linear_model.LinearRegression(fit_intercept=False)
    regression_2sls_2.fit(X=X_hat, y=y)
    b_hat_2sls = regression_2sls_2.coef_
    u_hat_2sls = y - X * b_hat_2sls

    # obtaining variance
    Omega_hat_2sls = np.diag(np.power(u_hat_2sls.A1, 2))
    # I dont really know why I need .A1 here. However I could not diagonalise
    # otherwise. Thanks stackexchange!

    # S_hat = (t(Z) * Omega_hat_2sls * Z) / N_obs
    Var_b_2sls = (inv(t(X_hat) * X_hat) * t(X_hat) *
                  Omega_hat_2sls *
                  X_hat * inv(t(X_hat) * X_hat))

    # formatting results:
    result = (b_hat_ols[0, 0], b_hat_2sls[0, 0],
              Var_b_ols[0, 0], Var_b_2sls[0, 0], (pi.T * Z.T * Z * pi))
    # concentration parameter is divided by 1 - variance of v.
    
    return [np.around(i, 3) for i in result]
    # print('b_hat_ols:  ' + str(b_hat_ols))
    # print('b_hat_2sls: ' + str(b_hat_2sls))
    # print('Var_b_ols:  ' + str(Var_b_ols))
    # print('Var_b_2sls: ' + str(Var_b_2sls))


#%%

def monte_carlo(gamma, rho, pi_1, n_iterations, N_obs, sigma_sq_z, r):
    '''
    sigma_sq_z = variance of matrix Z
    gamma = covariance between u an Z - instrument validity
    rho = covariance between u and X - level of endogeneity
    r = number of instruments
    N_obs = number of observations
    pi_1 = true coefficient between x and z - instrument relevance
    n_iterations = number of iterations of Monte Carlo

    This function basically just takes 'iterations' function and runs it
    'n_iterations' times.
    This is done to obtain the Hausman statistics from every iteration,
    then stored in results_Hausman, along with concentation parameter.

    This function returns results of Hausman test along with parameters used
    in the Monte Carlo simulation.
    '''
    results_Hausman = np.zeros((n_iterations, 3))
    for i in range(n_iterations):
        (b_hat_ols, b_hat_2sls, Var_b_ols, Var_b_2sls, conc_par) = iteration(
            gamma, rho, pi_1, N_obs, sigma_sq_z, r)
        d_Var = Var_b_2sls - Var_b_ols
        if d_Var == 0:
            d_Var_psinv = 0
        else:
            d_Var_psinv = 1 / d_Var

        Hausman = ((b_hat_ols - b_hat_2sls)) ** 2 * d_Var_psinv
        results_Hausman[i, 0] = Hausman
        df = b_hat_2sls[np.abs(b_hat_2sls) < 1e8].size
        pval = stats.chi2.sf(Hausman, df)
        results_Hausman[i, 1] = pval
        results_Hausman[i, 2] = conc_par
    return results_Hausman, [gamma, rho, pi_1, n_iterations, N_obs, sigma_sq_z, r]


#%%


def outcome(gamma, rho, pi_1, n_iterations, N_obs, sigma_sq_z, r, histogram):
    '''
    sigma_sq_z = variance of matrix Z
    gamma = covariance between u an Z - instrument validity
    rho = covariance between u and X - level of endogeneity
    r = number of instruments
    N_obs = number of observations
    pi_1 = true coefficient between x and z - instrument relevance
    n_iterations = number of iterations of Monte Carlo.
    histogram = True or False - whether we would like to present histogram


    This function basically just takes montecarlo results and presents them
    in readable way.

    '''
    Hausman_result, parameters = monte_carlo(
        gamma, rho, pi_1, n_iterations, N_obs, sigma_sq_z, r)
    used_pars = pd.DataFrame.from_items([('parameters:', parameters)],
                                        orient='index',
                                        columns=['gamma', 'rho', 'pi_1',
                                                 'n_iterations', 'N_obs',
                                                 'sigma_sq_z', 'r'])
    print(used_pars)
    positive_res = len(
        np.where(Hausman_result[:, 0] > stats.chi2.ppf(0.95, 1))[0])

    print('\nThe Hausman test passed the critical value for p-value 0.05 ' + str(positive_res) +
          ' times out of ' + str(parameters[3]) + ' iterations')
    print('The average concentration parameter is ' +
          str(np.mean(Hausman_result[:, 2])))

    if histogram:
        plt.hist(Hausman_result[:, 0], bins=40)
        plt.show()


#%%

# random.seed must be in every cell where new numbers are generated
# as the cell structure somewhat changes the way program
# runs in jupyter notebook/spyder


np.random.seed(studentnumber1 + 10 * studentnumber2)


def results(i, parameter_table, histogram):
    '''
    i - index of the parameter_table
    parameter_table - defined below
    wrapper of the 'outcome' function, which better fits in the 'for' loop.

    '''
    return outcome(gamma=parameter_table.iloc[i, 0],
                   rho=parameter_table.iloc[i, 1],
                   pi_1=parameter_table.iloc[i, 2],
                   n_iterations=parameter_table.iloc[i, 3],
                   N_obs=parameter_table.iloc[i, 4],
                   sigma_sq_z=parameter_table.iloc[i, 5],
                   r=parameter_table.iloc[i, 6],
                   histogram=histogram)


parameter_table = pd.DataFrame.from_items(
    [
        (1, [0, 0, 0.5, 10000, 50, 2, 3]),
        (2, [0, 0, 0.5, 10000, 200, 2, 3]),
        (3, [0, 0, 0.5, 10000, 200, 2, 1]),
        (4, [0, 0, 0.5, 10000, 200, 2, 5]),
        (5, [0, 0, 0.1, 10000, 200, 2, 3]),
        (6, [0, 0.25, 0.1, 10000, 200, 2, 3]),
        (7, [0, 0.25, 0.25, 10000, 200, 2, 3]),
        (8, [0, 0.25, 0.5, 10000, 200, 2, 3]),
        (9, [0, 0.5, 0.1, 10000, 200, 2, 3]),
        (10, [0, 0.5, 0.25, 10000, 200, 2, 3]),
        (11, [0, 0.5, 0.5, 10000, 200, 2, 3]),
        (12, [0, 0.5, 0.38, 10000, 200, 2, 3]),
        (13, [0, 0.5, 0.38, 10000, 100, 2, 3]),
        (14, [0, 0.5, 0.38, 10000, 200, 2, 10]),
        (15, [0.1, 0.1, 0.5, 10000, 200, 2, 3]),
        (16, [0.2, 0.2, 0.5, 10000, 200, 2, 3]),
    ],
    orient='index',
    columns=['gamma', 'rho', 'pi_1',
             'n_iterations', 'N_obs',
             'sigma_sq_z', 'r'])


#############################
#############################
## Presentation of results ##
#############################
#############################

print(
    '__________________________________________________________________\n\n' +
    '                             SECTION 1\n' +
    '__________________________________________________________________\n'
)


print(
    'In this section we investigate properties of parameters other than \n' +
    'gamma and rho. We set gamma and rho to zero, therefore we have situation\n' +
    'with unbiased OLS'
)


for i in range(5):
    print(
        '======= test ' + str(i + 1) + ' ======='
    )
    results(i=i,
            parameter_table=parameter_table,
            histogram=False)

print(
    '__________________________________________________________________\n\n' +
    '                             SECTION 2\n' +
    '__________________________________________________________________\n'
)


print(
    'In this section we investigate when rho does not equal to zero.'
)

for i in range(5, 14):
    print(
        '======= test ' + str(i + 1) + ' ======='
    )
    results(i=i,
            parameter_table=parameter_table,
            histogram=False)


print(
    '__________________________________________________________________\n\n' +
    '                             SECTION 3\n' +
    '__________________________________________________________________\n'
)


print(
    'In this section we investigate when both rho and sigma do not equal to zero.'
)

for i in range(14, 16):
    print(
        '======= test ' + str(i + 1) + ' ======='
    )
    results(i=i,
            parameter_table=parameter_table,
            histogram=False)


# If you would like to run only one simulation - run this function.
# outcome(gamma=0.1,
#         rho=0.25,
#         pi_1=0.25,
#         n_iterations=10000,
#         N_obs=200,
#         sigma_sq_z=2,
#         r=3,
#         histogram=False)
