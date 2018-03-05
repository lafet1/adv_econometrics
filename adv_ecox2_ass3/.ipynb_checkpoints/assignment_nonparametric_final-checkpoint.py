# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 17:19:29 2018

@author: StepanAsus
"""

# %%

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
# from scipy.stats import truncnorm


#%%

np.random.seed(2110)

def gen_data(N):
  d = np.random.normal(loc=55, scale=15, size=N).round().astype(int)
  # In this part we are taking normal distribution, with mean 55 and scale 15,
  # as in the original distribution.
  # The original distribution, on the other hand has different kurtosis,
  # and therefore a littlebit more mass in the centre
  # We decided not to care about that,
  # as thaht would not change further analysis in any significant way.
  # Also, we only take integers

  # any values less then zero are set to zero.
  d[d < 0] = 0

  # to reproduce the distribution,
  # we have decided to assign ~20% of mark 25 to 34
  # ~40% of 26 -> 33
  # ~60% of 27 -> 32
  # ~80% of 28 -> 31
  # ~95% of 29 -> 30
  # We assume that cases, when n={1,2,3,4,5} points were given are as common as
  # cases, where n points were needed.

  for i in range(25, 30):
      index = i - 24
      share = int(N * (1.1 * index * 0.17))
      data = d[:share]
      data[data == i] = 35 - index
      e = data.tolist() + d[share:].tolist()

  d = np.array(e)
  d[d > 110] = 100

  # This part is here for smoothing the right end of the distribution.
  for i in range(101, 111):
      index = i - 101
      share = int(N * (1 - (index * 0.095)))
      data = d[:share]
      data[data == i] = 101 - index
      e = data.tolist() + d[share:].tolist()

  d= np.array(e).reshape((N, 1))
  d[d > 100] = 54

  return d

d = gen_data(330000)
plt.hist(d, bins=101, normed=True)
plt.show()


# %%

# density estimation

def density_est(x, midpoints=None, bandwidth=0):

  x = x.reshape((x.shape[0], 1))
  nr, nc = x.shape

  # some data preparation
  x = x[:, 0]
  std = np.std(x)
  iota = np.ones((nr, 1))

  # here we pick the midpoints
  if midpoints==None:
    nrbins = max(x) - min(x) + 1
    midpoints_used = np.zeros((nrbins, 1))
    midpoints_used = np.linspace(min(x), max(x), num=nrbins).reshape((nrbins, 1)) # we know our data
  else:
    nrbins, ncbins = midpoints.size
    midpoints_used = midpoints

  # here the bandwidth is decided
  if bandwidth > 0:
    bandwidth_used = bandwidth
  else:
    bandwidth_used = 1.059 * std * nr ** (- 1 / 5)

  refl = min(midpoints_used)
  k_pdf = np.zeros((nrbins, 1))

  for J in range(nrbins):
     print(J)
     Xb = midpoints_used[J, 0]
     # in this part we decide whether the points should be mirrored.
     # This is determined whether the bandwidth is taking into account boundary observations.
     # However, we decided to solve only  left side of the interval.
     if (Xb - refl) < bandwidth_used: # comparison of takenpoint from midpoints, with subtracted minimum, with the bandwidth
     # if the bandwidth is greater than this boundary point, its kernel is taking into account zero values and therefore is biased
     # this plays a huge role in our density estimation
     # therefore, we decided to use mirroring technique.
     # We determine a coefficient with which should be the given distribution multiplied with
     # if the point is on the boundary - c = 2
     # if the point is close to boundary, but not exactly - 1 < c < 2
     # if away from boundary - c = 1
       c = (bandwidth_used - (Xb - refl)) / bandwidth_used
     else:
       c = 0
     Z = (iota * Xb - x) / bandwidth_used
     KX = norm.pdf(Z, 0, 1) / bandwidth_used + c * norm.pdf(Z, 0, 1) / bandwidth_used
     k_pdf[J, 0] = np.mean(KX)


  return(midpoints_used, k_pdf, bandwidth_used)


'''

###############

## N = 10000 ##

###############

'''

#%%

# First, we generate data and check its distribution

np.random.seed(2110)
d = gen_data(10000)
plt.hist(d, bins=101, normed=True)
plt.show()

#%%

# PART ONE:
# in this part two distributions are created, as we have multimodal distribution.
# We take into account the distribution where people have failed,
# and distribution where people passed.
#  We estimate the density for each part.

#   FAILED
d_fail = d[d < 30]
gaus1 = density_est(d_fail)
plt.plot(gaus1[1])


#%%

#    PASSED

d_pass = d[d > 29]

########################################

### DO NOT RUN - TAKES A LOT OF TIME ###

########################################

gaus2 = density_est(d_pass)
plt.plot(gaus2[1])


#%%

### PART TWO
# Now, we take the two distributions together,
# reweight them by their number of observations,
# and plot them.

share = len(d_fail) / len(d)
gaus = np.vstack((gaus1[1] * share, gaus2[1] * (1 - share)))
plt.figure(figsize=(9, 5))
plt.plot(gaus)
plt.hist(d, bins=101, normed=True)

plt.show()

# plt.savefig('n10000.png')

'''

##############

## N = 2500 ##

##############

'''

#%%

#  PART ONE
np.random.seed(2110)
d = gen_data(2500)
plt.hist(d, bins=101, normed=True)
plt.show()

#%%

#   FAILED
d_fail = d[d < 30]
gaus1 = density_est(d_fail)
plt.plot(gaus1[1])

#%%

#   PASSED
d_pass = d[d > 29]
gaus2 = density_est(d_pass)
plt.plot(gaus2[1])

#%%

#   PART TWO
share = len(d_fail) / len(d)
gaus = np.vstack((gaus1[1] * share, gaus2[1] * (1 - share)))
plt.figure(figsize=(9, 5))
plt.plot(gaus)
plt.hist(d, bins=101, normed=True)
plt.show()

'''

#############

## N = 500 ##

#############

'''

#%%

# PART ONE
np.random.seed(29)
d = gen_data(500)
plt.hist(d, bins=101, normed=True)
plt.show()

#%%

#   FAILED
d_fail = d[d < 30]
gaus1 = density_est(d_fail)
plt.plot(gaus1[1])

#   PASSED
d_pass = d[d > 29]
gaus2 = density_est(d_pass)
plt.plot(gaus2[1])


#%%

# PART TWO
share = len(d_fail) / len(d)
gaus = np.vstack((gaus1[1] * share, gaus2[1] * (1 - share)))

plt.figure(figsize=(9, 5))
plt.plot(gaus)
plt.hist(d, bins=101, normed=True)


# plt.savefig('n500.png')







#%%



'''

def cv(x, h):


  start = time.time()

  K2 = np.zeros((len(x), len(h)))

  f_hat = np.zeros((10, len(h)))



  for j in range(len(h)):

    for i in range(len(x)):

      # gaussian is currently in use, uniform and epanechnikov are implemented and unused

      # gaussian is used in the density estimation



      xi = x[i]

      u = (xi - x) / h[j]

      K2a = norm.pdf(u, 0, 2)

      #K2a = (2 - abs(u)) / 4

      #K2a = 3 / 160 * (2 - abs(u)) ** 3 * (4 + 6 * abs(u) + abs(u) ** 2)

      K2[i, j] = sum(K2a[2 >= abs(u)])



    for k in range(10):

      x_cv = np.delete(x, [range(int(k * len(x) / 10), int((k + 1) * len(x) / 10))])

      u = (xi - x_cv) / h[j]

      K = norm.pdf(u, 0, 1)

      #K = (1 - np.delete(u, i) ** 2) * 3 / 4

      #K = np.ones((len(x) - 1, 1)) / 2

      K = sum(K[1 >= abs(u)])

      f_hat[k, j] = K / (h[j] * len(x_cv))



  end = time.time()

  print(end - start)

  return([np.sum(K2, axis=0) / (h * len(x) ** 2) - 2 * np.sum(f_hat, axis=0) / 10, h])

'''

