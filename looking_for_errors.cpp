#include <iostream>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <vector>
#include <stdexcept>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <slope/slope.h>
#include <stats.hpp>

// -----------------------------
// proxSortedL1
// -----------------------------
static int evaluateProx(double *y, double *lambda, double *x, size_t n, int *order) {
  double d;
  double *s = nullptr;
  double *w = nullptr;
  size_t *idx_i = nullptr;
  size_t *idx_j = nullptr;
  size_t i, j, k;
  int result = 0;
  
  s = (double*)std::malloc(sizeof(double) * n);
  w = (double*)std::malloc(sizeof(double) * n);
  idx_i = (size_t*)std::malloc(sizeof(size_t) * n);
  idx_j = (size_t*)std::malloc(sizeof(size_t) * n);
  
  if (s && w && idx_i && idx_j) {
    k = 0;
    for (i = 0; i < n; i++) {
      idx_i[k] = i;
      idx_j[k] = i;
      s[k] = y[i] - lambda[i];
      w[k] = s[k];
      
      while ((k > 0) && (w[k - 1] <= w[k])) {
        k--;
        idx_j[k] = i;
        s[k] += s[k + 1];
        w[k] = s[k] / (i - idx_i[k] + 1);
      }
      k++;
    }
    
    if (order == NULL) {
      for (j = 0; j < k; j++) {
        d = w[j];
        if (d < 0) d = 0;
        for (i = idx_i[j]; i <= idx_j[j]; i++) x[i] = d;
      }
    } else {
      for (j = 0; j < k; j++) {
        d = w[j];
        if (d < 0) d = 0;
        for (i = idx_i[j]; i <= idx_j[j]; i++) x[order[i]] = d;
      }
    }
  } else {
    result = -1;
  }
  
  if (s) std::free(s);
  if (w) std::free(w);
  if (idx_i) std::free(idx_i);
  if (idx_j) std::free(idx_j);
  
  return result;
}

// argsort decreasing by values (for Eigen::VectorXd)
static void argsort_desc(const Eigen::VectorXd& w, std::vector<int>& ord) {
    ord.resize(w.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&w](int a, int b) {
        return w(a) > w(b);
    });
}

// sign
static Eigen::VectorXd sign(const Eigen::VectorXd& x) {
    Eigen::VectorXd result = Eigen::VectorXd::Zero(x.size());
    result = (x.array() > 0.0).select(1.0, result);
    result = (x.array() < 0.0).select(-1.0, result);
    return result;
}

// prox step of SLOPE(lambda) norm from point y
static Eigen::VectorXd prox_sorted_L1_C(const Eigen::VectorXd& y, const Eigen::VectorXd& lambda) {
    const size_t n = y.size();
    Eigen::VectorXd x(n);
    std::vector<int> order;
    
    // Need abs(y) for argsort
    Eigen::VectorXd y_abs = y.array().abs();
    argsort_desc(y_abs, order);
    
    Eigen::VectorXd sign_y = sign(y);  // -1,0,1
    
    // Sort |y| ascending for prox
    Eigen::VectorXd y_sorted = y_abs;
    std::sort(y_sorted.data(), y_sorted.data() + n);
    
    Eigen::VectorXd lambda_copy = lambda;
    evaluateProx(y_sorted.data(), lambda_copy.data(), x.data(), n, nullptr);
    
    // Unsort + restore signs
    Eigen::VectorXd res(n);
    for (size_t k = 0; k < n; ++k) {
        res[order[k]] = sign_y[order[k]] * x[k];
    }
    return res;
}

// Creates a vector of weights (lambda) for SLOPE
static void create_lambda(Eigen::VectorXd& lam, int p, double FDR, bool BH) {
    Eigen::VectorXd h(p);
    if (BH) {
        for (int i = 0; i < p; ++i) {
            h(i) = 1.0 - (FDR * (i + 1) / (2.0 * p));
        }
    } else {
        double const_val = FDR * 1.0 / (2.0 * p);
        h.setConstant(1.0 - const_val);
    }
    for (int i = 0; i < p; ++i) {
        lam(i) = stats::qnorm(h(i));
    }
}

// Expectation of truncated gamma distribution
static double Rf_pgamma(double x, double shape, double scale, int lower_tail, int log_p) {
    double val = stats::pgamma(x, shape, scale);  // shape, scale params!
    
    if (!lower_tail) val = 1.0 - val;
    if (log_p) return std::log(val);
    
    return val;
}

static double EX_trunc_gamma(double a, double b) {
    // E[Gamma(a,b) | X <= 1] = a/b * P(X<=1 | shape=a+1) / P(X<=1 | shape=a)
    double log_p1 = Rf_pgamma(1.0, a + 1.0, 1.0 / b, 1, 1);
    double log_p2 = Rf_pgamma(1.0, a,     1.0 / b, 1, 1);
    double c = std::exp(log_p1 - log_p2);
    c /= b;
    c *= a;
    return c;
}

// -----------------------------
// Eigen helpers
// -----------------------------
static inline bool is_finite(double x) {
  return std::isfinite(x);
}

// Center columns and normalise columns to unit L2 norm (approx arma::normalise default)
static void center_and_scale_eigen(Eigen::MatrixXd& X) {
  const int n = X.rows();
  const int p = X.cols();
  
  // center
  Eigen::RowVectorXd means = X.colwise().mean();
  X.rowwise() -= means;
  
  // normalise columns
  for (int j = 0; j < p; ++j) {
    double norm = X.col(j).norm();
    if (norm > 0) X.col(j) /= norm;
  }
}

// Scaling matrix X by weight vector w (divide each column by w[j])
static void div_X_by_w(Eigen::MatrixXd& X_div_w,
                       const Eigen::MatrixXd& X,
                       const Eigen::VectorXd& w,
                       int n, int p) {
    X_div_w.resize(n, p);
    for (int j = 0; j < p; ++j) {
        X_div_w.col(j) = X.col(j) / w(j);
    }
}

// Missing data imputation by column means
static void impute_mean(Eigen::MatrixXd& X, int n, int p) {
  for (int c = 0; c < p; ++c) {
    double colmean = 0.0;
    int non_na = 0;
    for (int r = 0; r < n; ++r) {
      if (is_finite(X(r, c))) {
        colmean += X(r, c);
        non_na += 1;
      }
    }
    if (non_na == 0) continue;
    colmean /= non_na;
    for (int r = 0; r < n; ++r) {
      if (!is_finite(X(r, c))) X(r, c) = colmean;
    }
  }
}

// Ledoit-Wolf-ish linear shrinkage covariance (ported line-by-line)
static void linshrink_cov(const Eigen::MatrixXd& X,
                          Eigen::MatrixXd& S,
                          int n, int p) {
  // means per column
  Eigen::RowVectorXd means = X.colwise().sum() / (double)n;
  
  // S = X'X then centered
  S = X.transpose() * X;
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      S(i, j) -= n * means(i) * means(j);
      S(i, j) /= (double)n;
    }
  }
  
  double m = S.trace() / (double)p;
  double d2 = S.norm();              // Frobenius
  double b_bar2 = d2 * d2;
  
  d2 = (b_bar2 - p * m * m) / (double)p;
  b_bar2 *= n;
  
  double prod;
  double sum_prod;
  
  for (int i = 0; i < p; ++i) {
    for (int j = 0; j < p; ++j) {
      sum_prod = 0.0;
      for (int k = 0; k < n; ++k) {
        prod = (X(k, i) - means(i)) * (X(k, j) - means(j));
        b_bar2 += prod * prod;
        sum_prod += prod;
      }
      b_bar2 -= 2.0 * S(i, j) * sum_prod;
    }
  }
  
  b_bar2 /= (p * std::pow(n - 1.0, 2.0));
  double b2 = (b_bar2 < d2) ? b_bar2 : d2;
  double a2 = d2 - b2;
  
  S *= (a2 / d2);
  m *= (b2 / d2);
  for (int i = 0; i < p; ++i) S(i, i) += m;
}

static Eigen::VectorXd slope_libslope_fit_quadratic(
    Eigen::MatrixXd& X,
    const Eigen::VectorXd& y,
    const Eigen::VectorXd& lambda_sorted_decreasing,
    double alpha = 1
) {
  const int n = X.rows();
  const int p = X.cols();
  
  slope::Slope model;

 // model.setNormalization("none");
  
  // Loss: quadratic (OLS)
  model.setLoss("quadratic");
  
  Eigen::MatrixXd ymat(n, 1);
  ymat.col(0) = y;
  
  Eigen::ArrayXd lam(p);
  for (int j = 0; j < p; ++j) {
    lam(j) = lambda_sorted_decreasing[j];
  }
  
  // std::cout << "X: " << X << "\n"
  //          << "ymat: " << ymat << "\n"
  //          << "alpha: " << alpha << "\n"
  //          << "lam: " << lam << "\n";
  
  // fit
  auto fit = model.fit(X, ymat, alpha, lam);
  
  std::cerr << "Intercept: " << fit.getIntercepts() << std::endl;

  auto B = fit.getCoefs();
  
  Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
  for (int k = 0; k < B.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(B, k); it; ++it) {
      beta(it.row()) = it.value();
    }
  }
  
  return beta;
}

// -----------------------------
// Imputation routines (Eigen-only)
// -----------------------------
static void impute_row_advance(
    const Eigen::VectorXd& beta,
    Eigen::MatrixXd& X,
    const Eigen::VectorXd& Y,
    const Eigen::MatrixXd& S,
    double sigma_sq,
    const std::vector<std::vector<bool>>& XisFin,  // <- Changed
    int n, int p,
    int row,
    const std::vector<int>& nanCols,
    const Eigen::VectorXd& m,
    const Eigen::VectorXd& tau_sq
)
{
    const int l = static_cast<int>(nanCols.size());
    if (l == 0) return;

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(l, l);
    Eigen::VectorXd u = Eigen::VectorXd::Zero(l);

    double r = Y(row);
    int u_ind = 0;

    for (int i = 0; i < p; ++i) {
        if (!XisFin[row][i]) {
            for (int j = 0; j < p; ++j) {
                if (XisFin[row][j]) {
                    u(u_ind) += X(row, j) * S(j, i);
                }
            }
            ++u_ind;
        } else {
            r -= beta(i) * X(row, i);
        }
    }

    for (int i = 0; i < l; ++i) {
        for (int j = 0; j < l; ++j) {
            if (i == j) {
                A(i, j) = 1.0;
            } else {
                int s = nanCols[i];
                int t = nanCols[j];
                A(i, j) = (beta(s) * beta(t) / sigma_sq + S(s, t)) / tau_sq(s);
            }
        }
    }

    Eigen::VectorXd b = Eigen::VectorXd::Zero(l);
    for (int i = 0; i < l; ++i) {
        int t = nanCols[i];
        b(i) = ((r * beta(t)) / sigma_sq + m(t) - u(i)) / tau_sq(t);
    }

    // solve(A, b)
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() != Eigen::Success) return;
    Eigen::VectorXd sol = ldlt.solve(b);

    for (int i = 0; i < l; ++i) {
        int t = nanCols[i];
        X(row, t) = sol(i);
    }
}

static void impute_advance(
    const Eigen::VectorXd& beta,
    Eigen::MatrixXd& X,
    const Eigen::VectorXd& Y,
    const Eigen::MatrixXd& S,
    double sigma_sq,
    int n, int p,
    const Eigen::VectorXd& mu,
    const std::vector<std::vector<bool>>& XisFin,      // changed
    const std::vector<int>& anyNanXrows,
    const std::vector<std::vector<int>>& nanIndInRows
) {
    Eigen::VectorXd tau_sq = (beta.array().square() / sigma_sq).matrix();
    for (int i = 0; i < p; ++i) tau_sq(i) = tau_sq(i) + S(i, i);
    
    Eigen::VectorXd m = Eigen::VectorXd::Zero(p);
    for (int i = 0; i < p; ++i) {
        for (int j = 0; j < p; ++j) {
            m(i) += mu(j) * S(i, j);
        }
    }
    
    for (int idx = 0; idx < static_cast<int>(anyNanXrows.size()); ++idx) {
        int row = anyNanXrows[idx];
        impute_row_advance(beta, X, Y, S, sigma_sq, XisFin, n, p, row,
                           nanIndInRows[idx], m, tau_sq);
    }
}

// gamma mean update
static void gamma_mean_update(
    const Eigen::VectorXd& abs_beta_ord,
    const Eigen::VectorXd& lambda,
    int p,
    Eigen::VectorXd& gamma_h
) {
    double bl_sum = lambda(0) * abs_beta_ord(0);
    double bl_mean;
    int equals = 1;

    for (int i = 1; i < p; ++i) {
        if (abs_beta_ord(i) == abs_beta_ord(i - 1)) {
            bl_sum += lambda(i) * abs_beta_ord(i);
            equals += 1;
        } else {
            bl_mean = bl_sum / equals;
            for (int j = i - equals; j < i; ++j) gamma_h(j) = bl_mean;
            bl_sum = lambda(i) * abs_beta_ord(i);
            equals = 1;
        }
    }

    bl_mean = bl_sum / equals;
    for (int j = p - equals; j < p; ++j) gamma_h(j) = bl_mean;
}

// -----------------------------
// Exported functions
// -----------------------------

Eigen::MatrixXd Center_and_scale(const Eigen::MatrixXd& Xr) {
    Eigen::MatrixXd X = Xr;
    center_and_scale_eigen(X);
    return X;
}

// Helper: sum(x != 0)
double sum_nz(const Eigen::VectorXd& x) {
    double s = 0.0;
    for (int i = 0; i < x.size(); ++i)
        s += (x(i) != 0.0) ? 1.0 : 0.0;
    return s;
}

// Helper: sum(abs(x))
double sum_abs(const Eigen::VectorXd& x) {
    return x.array().abs().sum();
}

// ---------------------
// SLOBE ADMM function
// ---------------------------

struct SLOBEResult {
    Eigen::VectorXd coefficients;
    double sigma;
    double theta;
    double c;
    Eigen::VectorXd w;
    bool converged;
    Eigen::MatrixXd X;
    Eigen::MatrixXd Sigma;
    Eigen::VectorXd mu;
    Eigen::VectorXd lambda;
};

SLOBEResult SLOBE_ADMM_approx_missing(
    const Eigen::VectorXd& start,
    const Eigen::MatrixXd& Xmis_r,
    const Eigen::MatrixXd& Xinit,
    const Eigen::VectorXd& Y_r,
    double a_prior,
    double b_prior,
    const Eigen::MatrixXd& Covmat_r,
    double sigma = 1.0,
    double FDR = 0.05,
    double tol = 1e-4,
    bool known_sigma = false,
    int max_iter = 100,
    bool verbose = false,
    bool BH = true,
    bool known_cov = false
) {
    const int p = static_cast<int>(start.size());
    const int n = static_cast<int>(Y_r.size());

    Eigen::VectorXd beta = start;
    Eigen::VectorXd beta_new(p);
    Eigen::VectorXd beta_e = beta;
    Eigen::VectorXd w = Eigen::VectorXd::Ones(p);
    Eigen::VectorXd w_e = Eigen::VectorXd::Ones(p);

    Eigen::VectorXd wbeta(p);
    Eigen::VectorXd gamma(p);
    Eigen::VectorXd gamma_h(p);
    Eigen::VectorXd lambda_sigma(p);

    std::vector<int> order(p);
    Eigen::MatrixXd X_div_w(n, p);

    double error = 0.0;
    double swlambda = 0.0;
    double RSS = 0.0;
    double sigma_sq = 1.0;

    // Missingness map from Xmis
    const Eigen::MatrixXd& Xmis = Xmis_r;

    std::vector<std::vector<bool>> XisFin(n, std::vector<bool>(p));
    std::vector<int> anyNanXrows;
    std::vector<std::vector<int>> nanIndicesInRow;

    for (int i = 0; i < n; ++i) {
        bool anyNan = false;
        std::vector<int> nanInd;
        for (int j = 0; j < p; ++j) {
            XisFin[i][j] = is_finite(Xmis(i, j));
            if (!XisFin[i][j]) {
                nanInd.push_back(j);
                anyNan = true;
            }
        }
        if (anyNan) {
            anyNanXrows.push_back(i);
            nanIndicesInRow.push_back(nanInd);
        }
    }

    // X starts from Xinit, then center/scale
    Eigen::MatrixXd X = Xinit;
    center_and_scale_eigen(X);

    // Sigma + precision S
    Eigen::MatrixXd Sigma(p, p);
    if (!known_cov) {
        linshrink_cov(X, Sigma, n, p);
    } else {
        Sigma = Covmat_r;
    }

    Eigen::LLT<Eigen::MatrixXd> llt(Sigma);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("Cholesky failed for Sigma.");
    }
    Eigen::MatrixXd S = llt.solve(Eigen::MatrixXd::Identity(p, p));

    // mu (column means)
    Eigen::VectorXd mu = X.colwise().mean().transpose();

    // lambda
    Eigen::VectorXd lambda(p);
    create_lambda(lambda, p, FDR, BH);

    // init sigma, c, theta, gamma
    double sstart = sum_nz(start);
    double pstart = (sstart > 0.0) ? sstart : 1.0;

    Eigen::VectorXd Y = Y_r;

    if (!known_sigma) {
        sigma = std::sqrt((X * beta_e - Y).squaredNorm() / (n - pstart));
    }

    lambda_sigma = lambda * sigma;

    argsort_desc(beta, order);

    double c = 0.0;
    if (sstart > 0.0) {
        double h = (sstart + 1.0) / sum_abs(beta) * (sigma / lambda(p - 1));
        c = (h < 1.0) ? h : 1.0;
    } else {
        c = 1.0;
    }

    double theta = (sstart + a_prior) / (a_prior + b_prior + p);

    // compute gamma_h on sorted abs coeffs
    Eigen::VectorXd beta_abs = beta.array().abs().matrix();
    Eigen::VectorXd beta_abs_sorted = beta_abs;
    Eigen::VectorXd beta_abs_ord = beta_abs;
    std::vector<int> order_abs;
    argsort_desc(beta_abs_ord, order_abs);
    std::sort(beta_abs_sorted.data(), beta_abs_sorted.data() + p, std::greater<double>());
    gamma_mean_update(beta_abs_sorted, lambda, p, gamma_h);
    gamma_h = gamma_h * (c - 1.0) / sigma;
    gamma_h = (theta * c) / (theta * c + (1.0 - theta) * gamma_h.array().exp()).array();
    for (int i = 0; i < p; ++i) {
        gamma( order[i]  ) = gamma_h[i];
    }

    bool converged = false;
    int iter = 0;

    while (iter < max_iter) {
        if (verbose) {
            std::cout << "Iteration: " << iter << "/" << max_iter << std::endl;
        }

        w = Eigen::VectorXd::Ones(p) - (1.0 - c) * gamma;
        w_e = w;

        div_X_by_w(X_div_w, X, w_e, n, p);

        Eigen::VectorXd lambda_sorted = lambda_sigma;
        std::sort(lambda_sorted.data(), lambda_sorted.data() + p, std::greater<double>());

        Eigen::VectorXd beta_hat = slope_libslope_fit_quadratic(X_div_w, Y, lambda_sorted, 1);

        int zeros = (beta_hat.array() == 0.0).count();
        std::cerr << "Number of zeros: " << zeros << std::endl;

        Eigen::VectorXd beta_hat_alpha = slope_libslope_fit_quadratic(X_div_w, Y, lambda_sorted, 0.005);

        int zeros_alpha = (beta_hat_alpha.array() == 0.0).count();
        std::cerr << "Number of zeros: " << zeros_alpha << std::endl;

        wbeta = beta_hat.cwiseAbs();
        argsort_desc(wbeta, order);

        // unweight
        for (int i = 0; i < p; ++i) beta_hat(i) /= w_e(i);
        beta_new = beta_hat;

        if (!known_sigma) {
            RSS = (X * beta_hat - Y).squaredNorm();
            Eigen::VectorXd wbeta_sorted = wbeta;
            std::sort(wbeta_sorted.data(), wbeta_sorted.data() + p, std::greater<double>());
            swlambda = (wbeta_sorted.array() * lambda.array()).sum();
            sigma = (swlambda + std::sqrt(swlambda * swlambda + 4.0 * (n + 2.0) * RSS))
                    / (2.0 * (n + 2.0));
        }

        lambda_sigma = lambda * sigma;
        sigma_sq     = sigma * sigma;

        if (!known_cov) {
            linshrink_cov(X, Sigma, n, p);
            Eigen::LLT<Eigen::MatrixXd> llt2(Sigma);
            if (llt2.info() != Eigen::Success) {
                throw std::runtime_error("Cholesky failed for Sigma (iter).");
            }
            S = llt2.solve(Eigen::MatrixXd::Identity(p, p));
        }

        mu = X.colwise().mean().transpose();

        impute_advance(beta_hat, X, Y, S, sigma_sq, n, p, mu, XisFin, anyNanXrows, nanIndicesInRow);
        center_and_scale_eigen(X);

        Eigen::VectorXd beta_new_abs = beta_new.array().abs().matrix();
        Eigen::VectorXd beta_new_abs_ord = beta_new_abs;
        std::vector<int> order_new;
        argsort_desc(beta_new_abs_ord, order_new);
        Eigen::VectorXd beta_new_abs_sorted = beta_new_abs;
        std::sort(beta_new_abs_sorted.data(), beta_new_abs_sorted.data() + p, std::greater<double>());
        gamma_mean_update(beta_new_abs_sorted, lambda, p, gamma_h);
        gamma_h = gamma_h * (c - 1.0) / sigma;
        gamma_h = (theta * c) / (theta * c + (1.0 - theta) * gamma_h.array().exp()).array();
        for (int i = 0; i < p; ++i) {
            gamma( order[i]  ) = gamma_h[i];
        }

        double sum_gamma = gamma.sum();
        double b_sum = 0.0;
        for (int k = 0; k < p; ++k) {
            int idx = order[k];
            b_sum += gamma[idx] * std::fabs(beta_new[idx]) * lambda[k];
        }
        b_sum /= sigma;

        if (sum_gamma > 0.0) {
            if (b_sum > 0.0) {
                c = EX_trunc_gamma(sum_gamma, b_sum);
            } else {
                c = (sum_gamma + 1.0) / (sum_gamma + 2.0);
            }
        } else {
            c = 0.5;
        }

        theta = (sum_gamma + a_prior) / (p + a_prior + b_prior);

        error = (beta - beta_new).array().abs().sum();
        if (error < tol) {
            converged = true;
            break;
        }

        if (verbose) {
            std::cout << "Error = " << error << " sigma = " << sigma
                      << " theta = " << theta << " c = " << c << std::endl;
        }

        beta = beta_new;
        ++iter;
    }

    SLOBEResult result;
    result.coefficients = beta;
    result.sigma        = sigma;
    result.theta        = theta;
    result.c            = c;
    result.w            = w;
    result.converged    = converged;
    result.X            = X;
    result.Sigma        = Sigma;
    result.mu           = mu;
    result.lambda       = lambda;

    return result;
}





























#include <fstream>
#include <sstream>

Eigen::MatrixXd read_csv(const std::string& path) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path);
    }

    std::vector<std::vector<double>> data;
    std::string line;
    
    std::getline(file, line); // skip header
    
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string val;
        while (std::getline(ss, val, ',')) {
            row.push_back(std::stod(val));
        }
        data.push_back(row);
    }
    
    int rows = data.size();
    int cols = data[0].size();
    Eigen::MatrixXd mat(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat(i, j) = data[i][j];
    return mat;
}

// For vectors (single column CSV):
Eigen::VectorXd read_csv_vec(const std::string& path) {
    Eigen::MatrixXd m = read_csv(path);
    return m.col(0);
}

















int main() {
    Eigen::MatrixXd Xmis  = read_csv("Xmis.csv");
    Eigen::MatrixXd Xinit = read_csv("Xinit.csv");
    Eigen::MatrixXd Covmat = read_csv("Covmat.csv");
    Eigen::VectorXd Y = read_csv_vec("Y.csv");
    Eigen::VectorXd start = read_csv_vec("start.csv");

    SLOBEResult res = SLOBE_ADMM_approx_missing(
        start, Xmis, Xinit, Y,
        1.0, 1.0, Covmat
    );

    //std::cout << "coefficients:\n" << res.coefficients << std::endl;
    std::cout << "sigma: "    << res.sigma     << std::endl;
    //std::cout << "converged: "<< res.converged << std::endl;

    return 0;
}
