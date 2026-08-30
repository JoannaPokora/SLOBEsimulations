calc_power <- function(beta, beta_est) {
  tp <- sum(beta[beta_est != 0] != 0)
  fn <- sum(beta[beta_est == 0] != 0)

  tp / (tp + fn)
}

calc_fdr <- function(beta, beta_est) {
  fp <- sum(beta[beta_est != 0] == 0)
  disc <- sum(beta_est != 0)
  
  fp / max(disc, 1)
}

calc_mse <- function(beta, beta_est) {
  sum((beta_est - beta)^2) / sum(beta^2)
}

calc_msp <- function(beta, beta_est, X) {
  if (is.null(beta_est) || length(beta_est) != length(beta)) {
    return(NA)
  }
  
  Xbeta <- X %*% beta
  sum((X %*% beta_est - Xbeta)^2) / sum(Xbeta^2)
}
