library(dplyr)
library(tidyr)
library(parallel)
library(pbapply)
library(ggplot2)
library(paletteer)

source("eval_funs.R")

rep_num <- 200

signals_num_lst <- list(c(3, 6, 10, 12, 15),
                        c(50, 100, 120, 150))

settings <- list(
  n = 100,
  p = c(100, 300),
  signals_num = unique(unlist(signals_num_lst)),
  corr = c(0, 0.5),
  signals_strength = c(7, 10, 15)
)

simulations <- expand.grid(settings) %>%
  rowwise() %>%
  filter({
    p_case <- p == settings$p
    signals_num %in% signals_num_lst[p_case][[1]]
  }) %>%
  ungroup() %>%
  mutate(case = 1:n())

# parallel computation
cl <- makeCluster(detectCores() - 1, outfile = "")

parallel::clusterEvalQ(cl, {
  library(slobe)
  library(glmnet)
  library(SLOPE)
  library(BhGLM)
  
  NULL
})

coefs <- lapply(1:nrow(simulations), function(row) {
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
      0,
      response = "binomial"
    )
    
    # new slobe
    coef_slobe <- slobe(train$X, train$y, family = "binomial")$coefficients[-1]
    
    # cvLASSO
    tryCatch({
      obj_cvLASSO <- cv.glmnet(train$X, train$y, family = "binomial",
                               type.measure = "auc",
                               standardize=FALSE, intercept=FALSE)
      scale_cvLASSO <- obj_cvLASSO$lambda.min
      coef_cvLASSO <- as.numeric(coef(obj_cvLASSO, s="lambda.min"))[-1]
      },
      error = function(e) list("error")
    )

    # cvSLOPE
    tryCatch({
      tune <- trainSLOPE(train$X, train$y, family = "binomial", q = 0.05,
                         lambda = "bh", scale = "none", measure = "auc",
                         intercept=FALSE)
      scale_SLOPE <- tune$optima$sigma
      coef_SLOPE <-
        tune$model$coefficients[,,tune$model$sigma == tune$optima$sigma]
    },
    error = function(e) list("error")
    )
    
    # ssl
    tryCatch({
      f1 <- glmNet(train$X, train$y, family = "binomial", ncv = 1) 
      ps <- f1$prior.scale; ps
      scale_sll <- ps
      ss <- c(ps, 0.5)
      ssl_res <- bmlasso(train$X, train$y, family = "binomial",
                         ss = ss, alpha = 1)
      coef_ssl <- as.numeric(ssl_res$beta)
    },
    error = function(e) list("error")
    )
    
    list(
      case = row,
      train = train,
      coef_slobe = coef_slobe,
      coef_cvLASSO = coef_cvLASSO,
      coef_SLOPE = coef_SLOPE,
      coef_SSL = coef_ssl
    )
  }, params = params, rep_num = rep_num, cl = cl)
})

stopCluster(cl)

res_df <- dplyr::bind_rows(
  lapply(coefs, function(case) {
    dplyr::bind_rows(
      lapply(case, function(rep_res) {
        tibble::tibble(
          case = rep_res$case,
          train = list(rep_res$train),
          SLOBE = list(rep_res$coef_slobe),
          cvLASSO = list(rep_res$coef_cvLASSO),
          SLOPE = list(rep_res$coef_SLOPE),
          SSL = list(rep_res$coef_SSL)
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
    y = list(train$y),
    beta = list(train$beta)
  ) %>%
  ungroup() %>%
  dplyr::select(-train) %>%
  pivot_longer(
    cols = c(SLOBE, cvLASSO, SLOPE, SSL),
    names_to = "fun",
    values_to = "coef"
  ) %>%
  rowwise() %>%
  mutate(
    power = calc_power(beta, coef),
    FDR = calc_fdr(beta, coef),
    MSE = calc_mse(beta, coef),
    MSP = calc_msp(beta, coef, X)
  ) %>%
  na.omit() %>%
  group_by(case, fun) %>%
  summarise(across(c(power, FDR, MSE, MSP), mean)) %>%
  left_join(
    dplyr::select(simulations,
                  case, p, signals_num, corr, signals_strength),
    by = "case"
  )

p_lab <- function(p) {
  paste0("p == ", p)
}

corr_lab <- function(corr) {
  paste0("rho == ", corr)
}

strength_lab <- function(sign_str) {
  paste0("strength == ", sign_str)
}

create_plot <- function(res_df, type, filt_p = 100) {
  res_df <- filter(res_df, p == filt_p)
  
  plt <- ggplot(res_df, aes(x = signals_num, y = .data[[type]], color = fun)) +
    geom_line() +
    geom_point() +
    facet_wrap(
      ~ p + corr + signals_strength,
      nrow = 1, scales = "free_x",
      labeller = labeller(
        p = as_labeller(p_lab, label_parsed),
        corr = as_labeller(corr_lab, label_parsed),
        signals_strength = as_labeller(strength_lab, label_parsed)
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
  
  if (type == "FDR") {
    plt +
      geom_hline(yintercept = 0.05, linetype = "dashed")
  } else {
    plt
  }
}

create_plot(res_df, "power")
create_plot(res_df, "FDR")
create_plot(res_df, "MSE")
create_plot(res_df, "MSP")

create_plot(res_df, "power", filt_p = 300)
create_plot(res_df, "FDR", filt_p = 300)
create_plot(res_df, "MSE", filt_p = 300)
create_plot(res_df, "MSP", filt_p = 300)
