
set.seed(12)

library(MASS)

get_data <- function(corr_matr, beta, n, p_missing, n_rep){
  X <- mvrnorm(n, mu = rep(0, n), Sigma = corr_matr)
  prob <- 1 / (1 + exp(-X %*% beta))
  y <- rbinom(n, 1, prob)
  X_vals_n <- n * nrow(corr_matr)
  X[sample(1:X_vals_n, X_vals_n * p_missing)] <- NA
}

n_rep <- 200
n <- 100
p <- c(100, 500)
true_sign <- list("p=100" = c(5, 10, 15, 20),
                  "p=500" = seq(10, 60, by = 10))
p_missing <- c(0.1, 0.2, 0.3)
corr <- c(0, 0.5, 0.9)
sign_strength_const <- 1:4

sign_strength <- setNames(
  lapply(p, function(p_val) sign_strength_const * sqrt(2 * log(p_val))),
  paste0("p=", p)
)

corr_matr <- setNames(lapply(p, function(p_val){
  setNames(lapply(corr, function(corr_val){
    toeplitz(corr_val^(0:(p_val - 1)))
  }), paste0("corr=", corr))
}), paste0("p=", p))

beta <- setNames(mapply(function(sign_s_p, p_val, true_sign_p){
  setNames(lapply(sign_s_p, function(sign_s_val){
    setNames(lapply(true_sign_p, function(true_sign_num){
      b <- rep(0, p_val)
      b[sample(1:p_val, true_sign_num)] <- sign_s_val
      b
    }), paste0("true_sign=", true_sign))
  }), paste0("signal_strength=", sign_strength_const))
}, sign_strength, p, true_sign, SIMPLIFY = FALSE), paste0("p=", p))

