library(readr)
library(ggplot2)
set.seed(1000)

d <- read_table("Chapter7data.txt", col_names = FALSE)

y <- as.numeric(unlist(d[, 1]))
N <- length(y)
X <- as.matrix(cbind(rep(1, N), d[, 2:4]))
Xr <- as.matrix(cbind(rep(1, N), d[, 2], d[, 3] + d[, 4]))

b <- c(0, 0.05, 0.05, 0.05) # let us start optimization there
b0 <- c(0, 0.1, 0.1, 0.1)
lambda0 <- exp(X %*% b0)

# unrestricted model
fn <- function(theta){
  - sum(dpois(y, lambda = exp(X[, 1] * theta[1] + X[, 2] * theta[2] + 
                                X[, 3] * theta[3] + X[, 4] * theta[4]), log = T))
}
optim <- nlm(fn, theta <- b, hessian = T) # optimization by hand
b_est <- c(optim$estimate)

lambda <- exp(X %*% b_est)
A <- - optim$hessian
A1 <- solve(A)

Vb <- - A1
seb <- sqrt(diag(Vb))

# restricted model
fnr <- function(theta){
  - sum(dpois(y, lambda = exp(Xr[, 1] * theta[1] + Xr[, 2] * theta[2] + 
                                Xr[, 3] * theta[3]), log = T))
}
optimr <- nlm(fnr, theta <- b[1:3], hessian = T) # optimization by hand
b_estr <- c(optimr$estimate)

lambdar <- exp(Xr %*% b_estr)
Ar <- - optimr$hessian
A1r <- solve(Ar)

Vbr <- - A1r / N
sebr <- sqrt(diag(Vbr))


LR <-  2 * (optimr$minimum - optim$minimum) # the minus is not in front as the log-likelihood is nonnegative
pchisq(LR, 1)



# Ex 1

r1 <- c(0, 0, 1, - 1)
w30 <- (b_est[3] - b_est[4])^2 / (r1 %*% Vb %*% r1)
1 - pchisq(w30, 1) # p-value of our statistic

# Ex 2

r2 <- c(0, 0, 1 / b_est[4], - b_est[3] / (b_est[4])^2)
w40 <- (b_est[3] / b_est[4] - 1)^2 / (r2 %*% Vb %*% r2)
1 - pchisq(w40, 1) # our p-value is much higher by simple hypothesis reformulation

# Ex 3

# score vector
s <- t(X) %*% (y - lambdar)

# Information matrix
I <- t(X) %*% apply(X, 2, '*', lambdar)
LM <- t(s) %*% solve(I) %*% s
1 - pchisq(LM, 1) # p-value, pretty much identical to w30

# Ex 4

# simulation of Wald test statistic
monte_carlo <- function(R){
  LR <- c() # for storing replication results
  w30 <- c()
  w40 <- c()
  LM <- c()
  
  for (i in 1:R){
    # data generation
    y <- rpois(N, lambda0)
    
    # unrestricted model
    fn <- function(theta){
      - sum(dpois(y, lambda = exp(X[, 1] * theta[1] + X[, 2] * theta[2] + 
                                    X[, 3] * theta[3] + X[, 4] * theta[4]), log = T))
    }
    optim <- nlm(fn, theta <- b, hessian = T)
    b_est <- c(optim$estimate)
    lambda <- exp(X * optim$estimate)
    A <- - optim$hessian
    A1 <- solve(A)
    VbML <- - A1
    
    # restricted model
    fnr <- function(theta){
      - sum(dpois(y, lambda = exp(Xr[, 1] * theta[1] + Xr[, 2] * theta[2] + 
                                    Xr[, 3] * theta[3]), log = T))
    }
    optimr <- nlm(fnr, theta <- b[1:3], hessian = T) # optimization by hand
    b_estr <- c(optimr$estimate)
    lambdar <- exp(Xr %*% optimr$estimate)
    Ar <- - optimr$hessian
    A1r <- solve(Ar)
    VbMLr <- - A1r
    
    # likelihood
    LR[i] <- 2 * (optimr$minimum - optim$minimum)
    # wald 1
    w30[i] <- (b_est[3] - b_est[4])^2 / (r1 %*% VbML %*% r1)
    # wald 2
    r2 <- c(0, 0, 1 / b_est[4], - b_est[3] / (b_est[4])^2)
    w40[i] <- (b_est[3] / b_est[4] - 1)^2 / (r2 %*% VbML %*% r2)
    # LM
    s <- t(X) %*% (y - lambdar)
    I <- t(X) %*% apply(X, 2, '*', lambdar)
    LM[i] <- t(s) %*% solve(I) %*% s
  }
  
  return(cbind(LR, w30, w40, LM))
}

results <- monte_carlo(1000)
c <- qchisq(0.95, 1)
stats <- apply(apply(results, 2, '>' , c), 2, mean)

ggplot(as.data.frame(results)) + 
  # geom_density(aes(x = results[, 1]), alpha = 0.5, fill = "indianred", color = "black") + 
  geom_density(aes(x = results[, 2]), alpha = 0.5, fill = "blue", color = "black") +
  geom_density(aes(x = results[, 3]), alpha = 0.5, fill = "green", color = "black") +
  # geom_density(aes(x = results[, 4]), alpha = 0.5, fill = "yellow", color = "black") + 
  coord_cartesian(xlim = c(0, 15))
  
# Ex 5
cv <- apply(results, 2, function(x) quantile(x, 0.95))

# Ex 6

# here we can also change the restrictions on the parameters, i.e. change our hypothesis
stats2 <- apply(apply(results, 2, '>' , cv), 2, mean)


