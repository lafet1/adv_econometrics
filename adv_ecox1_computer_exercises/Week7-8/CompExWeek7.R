##### Computer Exercise 7 #####

# Ex 1

# easy setup for the first exercise
beta <- 1
rho <- 0.5
r <- 3
pi_1 <- 1
pi <- c(pi_1, rep(0, r - 1))

gamma_1 <- 1 # heteroskedasticity parameter
gamma <- c(gamma_1, rep(0, r -1))
N <- 100

Z <- cbind(rnorm(N, 0, 1), rnorm(N, 0, 1), rnorm(N, 0, 1))
sigma <- exp(Z %*% gamma)

# matrices for results
R <- 200
b_2SLS <- matrix(sample(0, R, replace = T), R, 1)
b_GMM <- matrix(sample(0, R, replace = T), R, 1)
b_EGMM <- matrix(sample(0, R, replace = T), R, 1)
b_OLS <- matrix(sample(0, R, replace = T), R, 1)

for (i in 1:R){ # only a single loop to try it out, wrapper function in b
  u <- rnorm(N, 0, sigma)
  v <- (u / sigma) * rho + rnorm(N, 0, 1) * sqrt(1 - rho^2)
  x <- Z %*% pi + v
  y <- x %*% beta + u
  
  b_OLS[i] <- as.numeric(coef(lm(y ~ x))[2])
  b_2SLS[i] <- solve(t(x) %*% Z %*% solve(t(Z) %*% Z) %*% t(Z) %*% x) %*% t(x) %*% Z %*% solve(t(Z) %*% Z) %*%
    t(Z) %*% y
  e <- y - x %*% b_2SLS[i]

}


# Ex 2

# instead of setup we do this exercise by creating a wrapper function
monte_carlo <- function(gamma_1 = 0, rho = 0.5, N = 100, R = 200){
  
  # setup
  beta <- 1
  rho <- 0.5
  r <- 3
  pi_1 <- 1
  pi <- c(pi_1, rep(0, r - 1))
  
  gamma_1 <- 1 # heteroskedasticity parameter
  gamma <- c(gamma_1, rep(0, r -1))
  N <- 100
  
  Z <- cbind(rnorm(N, 0, 1), rnorm(N, 0, 1), rnorm(N, 0, 1))
  sigma <- exp(Z %*% gamma)
  
  # matrices for results
  R <- 200
  b_2SLS <- matrix(sample(0, R, replace = T), R, 1)
  b_GMM <- matrix(sample(0, R, replace = T), R, 1)
  b_EGMM <- matrix(sample(0, R, replace = T), R, 1)
  b_OLS <- matrix(sample(0, R, replace = T), R, 1)
  
  for (i in 1:R){
    u <- rnorm(N, 0, sigma)
    v <- (u / sigma) * rho + rnorm(N, 0, 1) * sqrt(1 - rho^2)
    x <- Z %*% pi + v
    y <- x %*% beta + u
    
    b_OLS[i] <- as.numeric(coef(lm(y ~ x))[2])
    b_2SLS[i] <- solve(t(x) %*% Z %*% solve(t(Z) %*% Z) %*% t(Z) %*% x) %*% t(x) %*% Z %*% solve(t(Z) %*% Z) %*%
      t(Z) %*% y
    e <- y - x %*% b_2SLS[i]
    
    S_hat <- matrix(rep(0, r*r),r, r)
    for (j in 1:N){
      u_sq <- (y[j] - x[j]*b_2SLS[j])^2
      S_hat <- S_hat + (u_sq * Z[j, ] %*% t(Z[j, ]))
    }
    S_hat <- S_hat * (1/N)
    
    b_GMM[i] <- (solve(t(x) %*% Z %*% solve(S_hat) %*% t(Z) %*% x) %*%
                   t(x) %*% Z %*% solve(S_hat) %*% t(Z)%*% y)
  }
  
  result <- list(b_OLS, b_2SLS, b_GMM)
  names(result) <- c("b_OLS", "b_2SLS", "b_GMM")
  return(result)
}

betas <- monte_carlo()

result <- function(betas){
  return(c(mean(betas) - beta, sd(betas), sqrt(mean((betas - beta)^2))))
}

res_matrix <- function(betas){
  row_names <- names(betas)
  len <- length(betas)
  width <- length(names(betas))
  results <- matrix(ncol = width, nrow = len)
  
  for (i in 1:len){
    results[i, ] <- result(betas[[i]])
  }
  colnames(results) <- c("bias", "Variance", "RMSE")
  rownames(results) <- row_names
  return(results)
}


res_matrix(monte_carlo(gamma_1 = 0, rho = 0.5))
res_matrix(monte_carlo(gamma_1 = 0.2, rho = 0.5))
res_matrix(monte_carlo(gamma_1 = 0.4, rho = 0.5))
res_matrix(monte_carlo(gamma_1 = 0.6, rho = 0.5))
res_matrix(monte_carlo(gamma_1 = 0.8, rho = 0.5))
res_matrix(monte_carlo(gamma_1 = 1, rho = 0.5))











