functions {
    vector gp_pred_rng(array[] real x2,
                       vector y1,
                       array[] real x1,
                       real alpha,
                       real rho,
                       real err,
                       real delta) {
      int N1 = rows(y1);
      int N2 = size(x2);
      vector[N2] f2;
      {
        matrix[N1, N1] L_K;
        vector[N1] K_div_y1;
        matrix[N1, N2] k_x1_x2;
        matrix[N1, N2] v_pred;
        vector[N2] f2_mu;
        matrix[N2, N2] cov_f2;
        matrix[N2, N2] diag_delta;
        matrix[N1, N1] K;
        K = gp_exp_quad_cov(x1, alpha, rho);
        for (n in 1:N1) {
            K[n, n] = K[n, n] + square(err);
        }
        L_K = cholesky_decompose(K);
        K_div_y1 = mdivide_left_tri_low(L_K, y1);
        K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
        k_x1_x2 = gp_exp_quad_cov(x1, x2, alpha, rho);
        f2_mu = (k_x1_x2' * K_div_y1);
        v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
        cov_f2 = gp_exp_quad_cov(x2, alpha, rho) - v_pred' * v_pred;
        diag_delta = diag_matrix(rep_vector(delta, N2));

        f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
      }
      return f2;
    }
}


data {
    int<lower=1> N_obs;                  // number of data points
    array[N_obs] real x;                 // independent quantities
    vector[N_obs] y;                     // dependent quantities
    int<lower=1> N_OOS;                  // number of prediction points
    array[N_OOS] real x_OOS;             // values to predict at
}


transformed data {
    // centre and scale data
    real x_mean = mean(x);
    real x_std = sd(x);
    real y_mean = mean(y);
    real y_std = sd(y);
    array[N_obs] real x_scaled;
    for(i in 1:N1){
        x_scaled[i] = (x[i] - x_mean) / x_std;
    }
    array[N_OOS] real x_OOS_scaled;
    for(i in 1:N_OOS){
        x_OOS_scaled[i] = (x_OOS[i] - x_mean) / x_std;
    }
    vector[N_obs] y_scaled = (y - y_mean) / y_std;

    vector[N_obs] mu = rep_vector(0, N_obs);
    real delta = 1e-9;
}


parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> err;
}


transformed parameters {
    array[3] real lprior;
    lprior[1] = inv_gamma_lpdf(rho | 3, 2);
    lprior[2] = normal_lpdf(alpha | 0, 4);
    lprior[3] = std_normal_lpdf(err);
}


model {
    target += sum(lprior);
    matrix[N_obs, N_obs] L_K;
    {
        matrix[N_obs, N_obs] K = gp_exp_quad_cov(x_scaled, alpha, rho);
        real sq_err = square(err);

        // diagonal elements
        for (n1 in 1:N_obs) {
            K[n1, n1] = K[n1, n1] + sq_err;
        }

        L_K = cholesky_decompose(K);
    }

    target += multi_normal_cholesky_lpdf(y_scaled | mu, L_K);
}


generated quantities {
    vector[N_obs] y_posterior;       // in-sample posterior predictive
    vector[N_OOS] y_OOS;             // out-sample predictions
    vector[N_obs] log_lik;

    {
        vector[N_obs] f1;       // in-sample posterior predictive mean (scaled)
        vector[N_obs] y_posterior_scaled;
        vector[N_OOS] f2;       // out-sample predictions (scaled)
        vector[N_OOS] y_OOS_scaled;

        f1 = gp_pred_rng(x_scaled, y_scaled, x_scaled, alpha, rho, err, delta);
        for (i in 1:N_obs) {
            y_posterior_scaled[i] = normal_rng(f1[i], err);
            log_lik[i] = multi_normal_cholesky_lpdf(y_scaled[i] | f1[i], err);
        }

        f2 = gp_pred_rng(x_OOS_scaled, y_scaled, x_scaled, alpha, rho, err, delta);
        for (i in 1:N_OOS) {
            y_OOS_scaled[i] = normal_rng(f2[i], err);
        }
    }
    // rescale data back
    y_posterior = y_posterior_scaled * y_std + y_mean;
    y_OOS = y_OOS_scaled * y_std + y_mean;
}