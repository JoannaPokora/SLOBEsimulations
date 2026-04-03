
library(SLOPE)
library(dplyr)
library(glmnet)
library(tidyr)
library(ggplot2)

norm_l2 <- function(X){
  apply(X, 2, function(x) x / sqrt(sum(x^2)))
}

get_lasso_beta <- function(X, Y){
  coefficients(cv.glmnet(X, Y))[2:(ncol(X) + 1), 1] 
}

calc_sigma <- function(X, Y, beta){
  sqrt(sum((X %*% beta - Y)^2) / (length(Y) - sum(beta != 0)))
}

create_lambda <- function(p, fdr = 0.05){
  h = 1 - (fdr * (1:p + 1) / (2 * p))
  qnorm(h)
}

calc_loss <- function(X, Y, beta, lambda, sigma){
  0.5 * sum((Y - X %*% beta)^2) +
    sigma * sum(sort(abs(beta), decreasing = TRUE) * lambda)
}

# skalowanie SLOPE:
# tylko przed
# przed i w trakcie
# tylko w trakcie SLOPE

# wyniki:
# liczba niezerowych współczynników
# loss

set.seed(12)

params <- list(
  n = c(100, 200),
  p = c(100, 200),
  scale = c("sd", "l2"),
  rep = 1:100
)
params_df <- expand.grid(params, stringsAsFactors = FALSE)

results <- params_df %>%
  group_by(p) %>%
  mutate(lambda_vec = list(create_lambda(p[1]))) %>%
  group_by(n, p, rep) %>%
  mutate(
    xy = list(SLOPE:::randomProblem(n[1], p[1], response = "gaussian")),
    X = list(as.matrix(xy[[1]]$x)),
    Y = list(xy[[1]]$y),
    lambda = list(create_lambda(p[1]))
  ) %>%
  rowwise() %>%
  mutate(
    X_cent_scale = list(
      if(scale == "sd")
        scale(X)
      else
        norm_l2(scale(X, scale = FALSE))
    ),
    lasso_beta = list(get_lasso_beta(X_cent_scale, Y)),
    sigma = calc_sigma(X_cent_scale, Y, lasso_beta),
    lambda_sigma = list(lambda * sigma),
    admm = list(ABSLOPE:::slope_admm(X_cent_scale, Y, lambda_sigma, p, 1)),
    libslope_scale_out =
      list(SLOPE(X_cent_scale, Y, center = "none", scale = "none",
                 lambda = lambda_sigma, alpha = 1)$coefficients[[1]][,1]),
    libslope_scale_out_and_in =
      list(SLOPE(X_cent_scale, Y, center = "mean", scale = scale,
                 lambda = lambda_sigma, alpha = 1)$coefficients[[1]][,1]),
    libslope_scale_in =
      list(SLOPE(X, Y, center = "mean", scale = scale,
                 lambda = lambda_sigma, alpha = 1)$coefficients[[1]][,1])
  )

models_info <- results %>%
  pivot_longer(cols = c(admm,
                        libslope_scale_out,
                        libslope_scale_out_and_in,
                        libslope_scale_in),
               names_to = "model",
               values_to = "coef") %>%
  rowwise() %>%
  mutate(
    nonzeros = sum(coef != 0),
    loss = calc_loss(X_cent_scale, Y, coef, lambda, sigma),
    n = paste0("n = ", n),
    p = paste0("p = ", p)
  )

ggplot(models_info, aes(x = model, y = nonzeros)) +
  geom_boxplot() +
  facet_wrap(scale ~ n ~ p, ncol = 4) +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(models_info, aes(x = model, y = loss)) +
  geom_boxplot() +
  facet_wrap(scale ~ n ~ p, ncol = 4, scales = "free_y") +
  theme_light() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
