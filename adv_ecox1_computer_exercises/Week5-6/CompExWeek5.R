###### CompEx Week 5/6 #####

Chapter5data <- read.table("~/UvA/Subjects/Advanced Econometrics 1/Computer Excercises/Week5-6/Chapter5data.txt",
                           quote="\"", comment.char="")
d <- Chapter5data


# Ex 1

# OLS
y <- d[, 1]
x <- matrix(d[, 2])
N <- length(y)
X <- cbind(matrix(rep(1, N)), x)
data <- cbind(y, X)
colnames(data) <- c("y", "x0", "x1")

mean(y)
sd(y)

b <- coefficients(lm(y ~ x))
e <- y - X %*% b
XX1 <- solve(t(X) %*% X)
Vb <- drop(((t(e) %*% e) / (N - 2))) * XX1
Vbw <- XX1 %*% t(X) %*% (e %*% t(e)) %*% X %*% XX1

seb <- sqrt(diag(Vb))
sebw <- sqrt(diag(Vbw))

# ML
fn <- function(theta){
  - sum(dexp(y, rate = exp(X[, 1] * theta[1] + X[, 2] * theta[2]), log = T))
}
optim <- nlm(fn, theta <- b, hessian = T) # optimization by hand

lambda <- exp(X %*% b)
h <- drop(t(y) %*% lambda) * X
A <- - optim$hessian

B <- t(h) %*% h / N
A1 <- solve(A)
VbML <- - A1 / N
VbQML <- A1 %*% B %*% A1 / N
sebML <- sqrt(diag(VbML))
sebQML <- sqrt(diag(VbQML))

library(bbmle)

# ML by built-in function
fitML <- mle2(y ~ dexp(rate = exp(x0 * theta1 + x1 * theta2)), start = list(theta1 = b[1], theta2 = b[2]), 
              data = as.data.frame(data))

# a

# i
# due to shape of distribution the OLS slope is other way around

# ii
# put OLS into QML and then show it is inconsistent


# Ex 2

# a
# simply show consistency by showing it satisfies the condition of ML and thus is QML

# b
# simply take a derivative

# c
fitNLS <- nls(y ~  exp(- (X[, 1] * theta1 + X[, 2] * theta2)), start = list(theta1 = b[1], theta2 = b[2]))


# Ex 3

R <- 1000
b_ML <- matrix(sample(0, R * 2, replace = T), R, 2)
b_NLS <- matrix(sample(0, R * 2, replace = T), R, 2)

for (i in 1:R){
  y1 <- rexp(N, rate = 1 / exp(x - 2))
  data1 <- cbind(y1, X)
  colnames(data1) <- c("y", "x0", "x1")
  b_ML[i, ] <- coef(mle2(y1 ~ dexp(rate = exp(x0 * theta1 + x1 * theta2)), start = list(theta1 = 2, theta2 = - 1), 
                    data = as.data.frame(data1)))
  b_NLS[i, ] <- coefficients(nls(y1 ~ exp( - (X[, 1] * theta1 + X[, 2] * theta2)), 
                                    start = list(theta1 = 2, theta2 = - 1)))
  
}

mean(b_ML[, 1])
mean(b_ML[, 2])
mean(b_NLS[, 1])
mean(b_NLS[, 2])

# NLS less precise and less efficient







