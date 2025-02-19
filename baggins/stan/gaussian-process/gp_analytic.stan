functions {
    vector gp_pred_rng(array[] real x2,
                       vector y1,
                       array[] real x1,
                       real alpha,
                       real rho,
                       real sigma,
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
            K[n, n] = K[n, n] + square(sigma);
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
    int<lower=1> N1;
    array[N1] real x1;
    vector[N1] y1;
    int<lower=1> N2;
    array[N2] real x2;
}


transformed data {
    // centre and scale data
    real x1_mean = mean(x1);
    real x1_std = sd(x1);
    real y1_mean = mean(y1);
    real y1_std = sd(y1);
    array[N1] real x1R;
    for(i in 1:N1){
        x1R[i] = (x1[i] - x1_mean) / x1_std;
    }
    array[N2] real x2R;
    for(i in 1:N2){
        x2R[i] = (x2[i] - x1_mean) / x1_std;
    }
    vector[N1] y1R = (y1 - y1_mean) / y1_std;

    vector[N1] mu = rep_vector(0, N1);
    real delta = 1e-9;
    int<lower=1> N = N1 + N2;
}


parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
}


transformed parameters {
    array[3] real lprior;
    lprior[1] = inv_gamma_lpdf(rho | 3, 2);
    lprior[2] = normal_lpdf(alpha | 0, 4);
    lprior[3] = std_normal_lpdf(sigma);
}


model {
    target += sum(lprior);
    matrix[N1, N1] L_K;
    {
        matrix[N1, N1] K = gp_exp_quad_cov(x1R, alpha, rho);
        real sq_sigma = square(sigma);

        // diagonal elements
        for (n1 in 1:N1) {
            K[n1, n1] = K[n1, n1] + sq_sigma;
        }

        L_K = cholesky_decompose(K);
    }

    target += multi_normal_cholesky_lpdf(y1R | mu, L_K);
}


generated quantities {
    vector[N1] f1;
    vector[N2] f2;
    vector[N] yR;

    f1 = gp_pred_rng(x1R, y1R, x1R, alpha, rho, sigma, delta);
    for (i in 1:N1) {
        yR[i] = normal_rng(f1[i], sigma);
    }

    f2 = gp_pred_rng(x2R, y1R, x1R, alpha, rho, sigma, delta);
    for (i in 1:N2) {
        yR[N1+i] = normal_rng(f2[i], sigma);
    }

    // rescale data back
    vector[N] y = yR * y1_std + y1_mean;
}