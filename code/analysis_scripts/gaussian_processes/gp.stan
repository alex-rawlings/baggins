// Gaussian process fitting following example in
// https://mc-stan.org/docs/2_19/stan-users-guide/fit-gp-section.html
data {
    // number of data points
    int<lower=1> N;
    // observed independent variable
    array[N] real<lower=0> theta;
    // observed dependent variable
    array[N] real <lower=0, upper=1> ecc;
}


transformed data {
    vector[N] mu = rep_vector(0, N);
}


parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
}


model {
    matrix[N, N] L_K;
    matrix[N, N] K = cov_exp_quad(theta, alpha, rho);
    real sq_sigma = square(sigma);

    // diagonal elements
    for(n in 1:N){
        K[n, n] = K[n, n] + sq_sigma;
    }

    L_K = cholesky_decompose(K);

    target += inv_gamma_lpdf(rho | 5, 5);
    target += std_normal_lpdf(alpha);
    target += std_normal_lpdf(sigma);

    target += multi_normal_cholesky_lpdf(ecc | mu, L_K);
}