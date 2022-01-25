################################################################################
## Filename: gradient-descent-lm.r
## Description: 
## Author: Helge Liebert
## Created: Di Aug 11 16:12:24 2020
## Last-Updated: Di Jan 25 14:51:26 2022
################################################################################

#================================= Model inputs ================================

# generate random data in which y is a noisy function of x
x1 <- runif(1000, -5, 5)
x2 <- runif(1000, -15, 15)
x3 <- runif(1000, -1, 7)
intercept <- rep(1, 1000)

## Outcome and matrix of regressors plus intercept

## univariate
y <- x1 + rnorm(1000) + 3
X <- as.matrix(cbind(intercept, x1))

## multivariate
## y <- 3*x1 + 0.5*x2 + 15*x3  + rnorm(1000) + 3
## X <- as.matrix(cbind(intercept, x1, x2, x3))


#================================== QR decomp ==================================

# Fit a linear model, QR-decomposition/Gram-Schmidt orthogonalization
b.qrd <- lm(y ~ X - 1)
## b.rd <- lm(y ~ x1)
## b.qrd <- lm(y ~ x1 + x2 + x3)
b.qrd

# Doing qrd manually and solving
## After rotation of `X` and `y`, solve upper triangular system `Rb = Q'y`
qrdecomp <- qr(X)
b.qrd <- backsolve(qrdecomp$qr, qr.qty(qrdecomp, y))
b.qrd


#===================================== SVD =====================================

## Moore-penrose Pseudoinverse
svdecomp <- svd(X)
str(svdecomp)
b.svd <- as.numeric(svdecomp$v %*% solve(diag(svdecomp$d)) %*% t(svdecomp$u) %*% y)
b.svd
b.svd <- as.numeric(svdecomp$v %*% (crossprod(svdecomp$u, y) / svdecomp$d)) ## the same
b.svd
b.svd <- MASS::ginv(X) %*% y ## function for Moore-Penrose Pseudo-Inverse
b.svd

## Moore-penrose Pseudoinverse, computed using eigenvalue decomposition of x'x
egd <- eigen(t(X) %*% X)
xtx.inv <- egd$vectors %*% solve(diag(egd$values)) %*% t(egd$vectors)
x.pseudo.inv <- xtx.inv %*% t(X)
b.svd <- x.pseudo.inv %*% y
b.svd

#=============================== Normal equations ==============================

## Matrix multiplication
b.ols <- solve(t(X) %*% X) %*% t(X) %*% y
b.ols
## More idiomatic
b.ols <- solve(crossprod(X)) %*% crossprod(X,y)
b.ols
b.ols <- solve(crossprod(X), crossprod(X,y))
b.ols


#=============================== Gradient descent ==============================

## squared error cost function
cost <- function(X, y, theta) {
  sum((X %*% theta - y)^2) / (2 * length(y))
}

## euclidean norm
eucnorm <- function(x) sqrt(sum(x^2))

## learning rate and iteration limit
alpha <- 0.01
niter <- 2000


## Batch gradient descent
## initialize coefficients
theta <- matrix(c(0, 0), nrow = 2) ## univariate
## theta <- matrix(c(0, 0), nrow = 4) ## multivariate
## keep history
cost_history <- double(niter)
theta_history <- list(niter)
## compute gradient and update
set.seed(42)
for (i in 1:niter) {
  error <- (X %*% theta) - y
  delta <- t(X) %*% error / length(y)
  theta <- theta - alpha * delta
  cost_history[i] <- cost(X, y, theta)
  theta_history[[i]] <- theta
  ## if ((i %% 100) == 0) print(theta)
}
print(theta)
print(niter)


## Batch gradient descent
## with a non-fixed number of iterations and stopping rule
## initialize coefficients
theta <- matrix(c(0, 0), nrow = 2) ## univariate
## theta <- matrix(c(0, 0), nrow = 4) ## multivariate
## keep history
cost_history <- c()
theta_history <- list()
## threshold and first iteration values
set.seed(42)
epsilon <- 10e-10
delta <- Inf
i = 0
## compute gradient and update
while (eucnorm(delta) > epsilon) {
  error <- (X %*% theta) - y
  delta <- t(X) %*% error / length(y)
  theta <- theta - alpha * delta
  cost_history <- c(cost_history, cost(X, y, theta))
  theta_history <- append(theta_history, list(theta))
  ## if ((i %% 100) == 0) print(theta)
  i <- i + 1
}
print(theta)
print(i)


#===================================== Plot ====================================

## compare estimates
plot(x1, y, col = rgb(0.2, 0.4, 0.6, 0.4),
     main = "Linear regression (by QR-decomp, normal equations or gradient descent)")
abline(b.qrd[1:2], col = "blue")
abline(b.ols[1:2], col = "green")
abline(theta[1:2], col = "red")

## plot data and converging fit
plot(x1, y, col = rgb(0.2, 0.4, 0.6, 0.4),
     main = "Linear regression by gradient descent")
for (i in c(1, 3, 6, 10, 14, seq(20, niter, by = 10))) {
  abline(coef = theta_history[[i]], col = rgb(0.8, 0, 0, 0.3))
}
abline(coef = theta, col = "blue")

## cost convergence
plot(cost_history, type = "l", col = "blue", lwd = 2,
     main = "Cost function", ylab = "cost", xlab = "Iterations")



#=================== Stochastic gradient descent, single obs ===================

## Computing the gradient over all data is expensive. Stochastic gradient
## descent uses only a single observation to compute the gradient, which is much
## quicker, and less prone to overshooting with a badly chosen step size. Other
## sgd variants use a small subsample of observations, or multiple subsamples in
## parallel ('mini-batch sgd').

theta <- matrix(c(0, 0), nrow = 2) ## univariate
## theta <- matrix(c(0, 0), nrow = 4) ## multivariate
cost_history <- c()
theta_history <- list()
set.seed(42)
for (i in 1:niter) {
  ## j <- sample(NROW(X), NCOL(X))
  j <- sample(NROW(X), 1)
  error <- (X[j, ] %*% theta) - y[j]
  ## delta <- t(X[j, ]) %*% error / length(y[j]) ## no need to transpose vector
  delta <- X[j, ] %*% error / length(y[j])
  theta <- theta - alpha * delta
  cost_history[i] <- cost(X[j, ], y[j], theta) ## cost function could be simplified
  theta_history[[i]] <- theta
}
print(theta)

## plot data and converging fit
plot(x1, y, col = rgb(0.2, 0.4, 0.6, 0.4),
     main = "Linear regression by stochastic gradient descent, single obs")
for (i in c(1, 3, 6, 10, 14, seq(20, niter, by = 10))) {
  abline(coef = theta_history[[i]], col = rgb(0.8, 0, 0, 0.3))
}
abline(coef = theta, col = "blue")

## cost convergence
plot(cost_history, type = "l", col = "blue", lwd = 2,
     main = "Cost function", ylab = "cost", xlab = "Iterations")


#================== Stochastic gradient descnet, single batch ==================

theta <- matrix(c(0, 0), nrow = 2) ## univariate
## theta <- matrix(c(0, 0), nrow = 4) ## multivariate
cost_history <- c()
theta_history <- list()
set.seed(42)
for (i in 1:niter) {
  select <- sample(NROW(X), 32)
  error <- (X[select, ] %*% theta) - y[select]
  delta <- t(X[select, ]) %*% error / length(y[select])
  theta <- theta - alpha * delta
  cost_history[i] <- cost(X[select, ], y[select], theta)
  theta_history[[i]] <- theta
}
print(theta)

## plot data and converging fit
plot(x1, y, col = rgb(0.2, 0.4, 0.6, 0.4),
     main = "Linear regression by stochastic gradient descent, mini batch")
for (i in c(1, 3, 6, 10, 14, seq(20, niter, by = 10))) {
  abline(coef = theta_history[[i]], col = rgb(0.8, 0, 0, 0.3))
}
abline(coef = theta, col = "blue")

## cost convergence
plot(cost_history, type = "l", col = "blue", lwd = 2,
     main = "Cost function", ylab = "cost", xlab = "Iterations")


#============================== More illustrations =============================

conv <- as.data.frame(cbind(t(sapply(theta_history, function(x) x[, 1])), cost = cost_history))
head(conv)

library("plot3D")
scatter3D(
  x = conv$intercept,
  y = conv$x1,
  z = conv$cost,
  xlab = "intercept",
  ylab = "slope",
  zlab = "cost (mse)",
  col = ramp.col(
    col = sort(RColorBrewer::brewer.pal(9, "Blues"), decreasing = F),
    n = length(unique(conv$cost))
  ),
  colkey = F,
  phi = 10,
  theta = 45,
  main = "Gradient Descent (3D View)"
)


#======== Stochastic gradient descent, mini batch w/ multiple batches =======

## ...


