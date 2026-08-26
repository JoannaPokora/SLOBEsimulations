library(grpSLOPE)

slope_admm <- function(A, b, z, u, lambda_seq, rho,
                       max_iter = 500, tol_infeas = 1e-8,
                       verbose = FALSE){
  M <- solve(crossprod(A) + diag(rho, ncol(A)))
  MtAb <- M %*% crossprod(A,b)
  lambda_seq_rho <- lambda_seq/rho
  z_new <- NULL
  for(iter in 1:max_iter){ #just until we do not choose some reasonable convergence criterion

    x <- MtAb + crossprod(M, (rho*(z - u)))
    z_new <- grpSLOPE::prox_sorted_L1(x = as.vector(x + u), lambda = lambda_seq_rho)
    u <- u + x - z_new

    dual_feasibility <- sqrt(sum((rho*(z_new-z))^2))
    primal_feasibility <- sqrt(sum((z_new - x)^2))

    z <- z_new

    if(verbose)
      message(sprintf("Iter %i\nprimal: %f\ndual: %f\n",
                      iter, primal_feasibility, dual_feasibility))

    if(dual_feasibility < tol_infeas & primal_feasibility < tol_infeas){
      break;
    }
  }
  return(list(x = x, z = z, u = u,
              primal_feasibility = primal_feasibility,
              dual_feasibility = dual_feasibility))
}





