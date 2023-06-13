// Gaussian process fitting following example "Pedictive inference with a GP" in
// https://mc-stan.org/docs/2_19/stan-users-guide/fit-gp-section.html
data {
    // number of data points
    int<lower=1> N;
    // observed independent variable
    array[N] real<lower=0> theta;
    // observed dependent variable
    vector<lower=0, upper=1>[N] ecc;
}


transformed data {
    real delta = 1e-9;
}


parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
    vector[N] eta;
}


transformed parameters {
    vector[N] f;
    {
        matrix[N, N] L_K;
        matrix[N, N] K = cov_exp_quad(theta, alpha, rho);
        // diagonal elements
        for(n in 1:N){
            K[n, n] = K[n, n] + delta;
        }
        L_K = cholesky_decompose(K);
        f = L_K * eta;
    }
}


model {
    target += inv_gamma_lpdf(rho | 5, 5);
    target += std_normal_lpdf(alpha);
    target += std_normal_lpdf(sigma);
    target += std_normal_lpdf(eta);

    target += normal_lpdf(ecc | f, sigma);
}


generated quantities {
    array[N] real y;
    y = normal_rng(f, sigma);
}