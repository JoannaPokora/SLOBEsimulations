library(dplyr)
library(tidyr)
library(parallel)
library(pbapply)
library(ggplot2)
library(paletteer)

source("eval_funs.R")

rep_num <- 200

settings <- list(
  n = 100,
  p = 100,
  signals_num = c(3, 6, 10, 12, 15),
  p_missing = c(0, 0.1),
  corr = c(0, 0.5),
  sign_strength_const = c(1.3, 3)
)

simulations <- expand.grid(settings) %>%
  mutate(case = 1:n()) %>%
  rowwise() %>%
  mutate(
    signals_strength = sign_strength_const * sqrt(2 * log(p))
  ) %>%
  ungroup()

# parallel computation
cl <- makeCluster(detectCores() - 1, outfile = "")

cpp_file <- normalizePath('abslope2021_source/SLOBE_cpp_missing.cpp')

parallel::clusterExport(cl, "cpp_file")

parallel::clusterEvalQ(cl, {
  library(slobe)
  library(Rcpp)
  library(SLOPE)
  library(truncdist)
  library(nlshrink)
  library(MASS)
  library(glmnet)
  library(missMDA)
  
  source("abslope2021_source/slope_admm.R")
  
  Rcpp::sourceCpp(
    cpp_file,
    rebuild = FALSE,
    verbose = FALSE
  )
  
  NULL
})

clusterExport(
  cl,
  c("ABSLOPE", "SLOBEC", "rescale", "rescale_all"),
  envir = globalenv()
)

trained_models <- lapply(1:nrow(simulations), function(row) {
  params <- simulations[row, ]
  
  cat("case", row, "/", nrow(simulations), "\n")
  
  pbapply::pblapply(1:rep_num, function(rep, params, rep_num) {
    set.seed((row - 1) * rep_num + rep)
    
    train <- slobe:::generate_data(
      params$n,
      params$p,
      params$signals_num,
      params$signals_strength,
      params$corr,
      params$p_missing
    )
    
    # a and b prior
    ab_prior <- 0.01 * params$n
    
    # BH lambda sequence
    lambda_bhq <- SLOPE:::create_lambda_bhq(params$n, params$p, 0.05)
    
    # Xmis and Xinit
    if (params$p_missing > 0) {
      Xmis <- train$X
      Xinit <- apply(Xmis, 2, function(col) {
        missing_vals <- is.na(col)
        
        if (any(missing_vals)) {
          col[missing_vals] <- mean(col, na.rm = TRUE)
        }
        
        col
      })
    } else {
      Xmis <- train$X
      Xinit <- train$X
    }
    
    Xinit <- slobe:::center_and_scale(Xinit)
    
    # lasso start
    lasso_init <- glmnet::cv.glmnet(
      Xinit, train$y, standardize = FALSE, intercept = FALSE)
    start <- stats::coefficients(lasso_init, s = "lambda.min")
    start <- start[2:(ncol(Xinit) + 1), 1]
    
    # new slobe
    t0 <- proc.time()[["elapsed"]]
    fit_slobe <- tryCatch(
      slobe(train$X, train$y, tol = 1e-3),
      error = function(e) list("error")
    )
    time_slobe <- proc.time()[["elapsed"]] - t0

    # ABSLOPE
    t0 <- proc.time()[["elapsed"]]
    fit_ABSLOPE <- tryCatch(
      ABSLOPE(train$X, train$y, lambda_bhq,
              ab_prior, ab_prior, impute = "mean", tol_em = 1e-3),
      error = function(e) list("error")
    )
    time_ABSLOPE <- proc.time()[["elapsed"]] - t0

    # old SLOBE
    t0 <- proc.time()[["elapsed"]]
    fit_SLOBE <- tryCatch(
      SLOBEC(start, Xmis, Xinit, train$y,
             ab_prior, ab_prior, tol = 1e-3),
      error = function(e) "error"
    )
    time_SLOBE <- proc.time()[["elapsed"]] - t0

    list(
      case = row,
      train = train,
      slobe = fit_slobe,
      ABSLOPE = fit_ABSLOPE,
      SLOBE = fit_SLOBE,
      time_slobe = time_slobe,
      time_ABSLOPE = time_ABSLOPE,
      time_SLOBE = time_SLOBE
    )
  }, params = params, rep_num = rep_num, cl = cl)
})

stopCluster(cl)

saveRDS(trained_models, "results/trained_models.RDS")

any(unlist(lapply(trained_models, function(x) {
  lapply(x, function(y) {
    y$ABSLOPE[[1]] == "error"
  })
})))

trained_models <- readRDS("results/trained_models.RDS")

res_df <- dplyr::bind_rows(
  lapply(trained_models, function(case) {
    dplyr::bind_rows(
      lapply(case, function(rep_res) {
        tibble::tibble(
          case = rep_res$case,
          train = list(rep_res$train),
          slobe = list(rep_res$slobe),
          ABSLOPE = list(rep_res$ABSLOPE),
          SLOBE = list(rep_res$SLOBE),
          time_slobe = rep_res$time_slobe,
          time_ABSLOPE = rep_res$time_ABSLOPE,
          time_SLOBE = rep_res$time_SLOBE
        )
      })
    )
  })
) %>%
  mutate(
    rep = rep(1:rep_num, nrow(simulations)), .after = 1,
  ) %>%
  rowwise() %>%
  mutate(
    X = list(train$X),
    X_nomis = list(train$X_nomis),
    y = list(train$y),
    beta = list(train$beta),
    coef_slobe = list(slobe$coefficients[-1]),
    coef_ABSLOPE = list(ABSLOPE$beta.new),
    coef_SLOBE = list(SLOBE$beta)
  ) %>%
  ungroup() %>%
  dplyr::select(-c(train, slobe, ABSLOPE, SLOBE)) %>%
  mutate(across(3:5, as.list)) %>%
  pivot_longer(
    cols = c(3:5, 10:12),
    names_to = c("type", "fun"),
    names_pattern = "(.*)_(A*B*.....)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = type,
    values_from = value
  ) %>%
  mutate(
    time = unlist(time),
    fun = case_when(
      fun == "slobe" ~ "new SLOBE",
      fun == "SLOBE" ~ "old SLOBE",
      .default = fun
    ),
    fun = factor(fun, levels = c("new SLOBE", "old SLOBE", "ABSLOPE"))
  ) %>%
  rowwise() %>%
  mutate(
    X_nomis = list(case_when(
      any(is.null(X_nomis)) ~ X,
      .default = X_nomis
    )),
    power = calc_power(beta, coef),
    FDR = calc_fdr(beta, coef),
    MSE = calc_mse(beta, coef),
    MSP = calc_msp(beta, coef, X_nomis)
  ) %>%
  na.omit() %>%
  group_by(case, fun) %>%
  summarise(across(c(time, power, FDR, MSE, MSP), mean)) %>%
  left_join(
    dplyr::select(simulations,
                  case, signals_num, p_missing, corr, sign_strength_const),
    by = "case"
  )


p_missing_lab <- function(p_missing) {
  paste0("'NA' == ", p_missing)
}

corr_lab <- function(corr) {
  paste0("rho == ", corr)
}

strength_lab <- function(sign_str) {
  paste(ifelse(sign_str == 1.3, "weak", "strong"), "signals")
}

create_plot <- function(res_df, type, filter_fun = NULL) {
  if (!is.null(filter_fun)) {
    res_df <- filter(res_df, fun != filter_fun)
  }
  ggplot(res_df, aes(x = signals_num, y = .data[[type]], color = fun)) +
    geom_line() +
    geom_point() +
    facet_wrap(
      ~ p_missing + corr + sign_strength_const,
      nrow = 2,
      labeller = labeller(
        p_missing = as_labeller(p_missing_lab, label_parsed),
        corr = as_labeller(corr_lab, label_parsed),
        sign_strength_const = strength_lab
      )
    ) +
    scale_colour_paletteer_d("ggthemes::Classic_10") +
    labs(x = "number of signals", y = type, colour = NULL) +
    theme_light() +
    theme(
      strip.background = element_blank(),
      strip.text = element_text(
        colour = "black",
        size = 9,
        margin = margin(t = 1, b = 2)
      )
    )
}

create_plot(res_df, "time")
create_plot(res_df, "power")
create_plot(res_df, "FDR")
create_plot(res_df, "MSE")
create_plot(res_df, "MSP")

create_plot(res_df, "time", "ABSLOPE")
create_plot(res_df, "power", "ABSLOPE")
create_plot(res_df, "FDR", "ABSLOPE")
create_plot(res_df, "MSE", "ABSLOPE")
create_plot(res_df, "MSP", "ABSLOPE")
