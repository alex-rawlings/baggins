// Gaussian process fitting following example "Pedictive inference with a GP" in
// https://mc-stan.org/docs/2_19/stan-users-guide/fit-gp-section.html
data {
    // number of data points to infer on
    int<lower=1> N1;
    // observed independent variable
    array[N1] real x1;
    // observed dependent variable
    vector[N1] y1;
    // number of data points to predict on
    int<lower=1> N2;
    // independent variable to predict on
    array[N2] real x2;
}


transformed data {
    real delta = 1e-9;
    int<lower=1> N = N1 + N2;
    array[N] real x = append_array(x1, x2);
}


parameters {
    real<lower=0> rho;
    real<lower=0> alpha;
    real<lower=0> sigma;
    vector[N] eta;
}


transformed parameters {
    array[3] real lprior;
    lprior[1] = inv_gamma_lpdf(rho | 2, 2);
    lprior[2] = normal_lpdf(alpha | 0, 4);
    lprior[3] = std_normal_lpdf(sigma);
    vector[N] f;
    {
        matrix[N, N] L_K;
        matrix[N, N] K = gp_matern32_cov(x, alpha, rho);
        // diagonal elements
        for(n in 1:N){
            K[n, n] = K[n, n] + delta;
        }
        L_K = cholesky_decompose(K);
        f = L_K * eta;
    }
}


model {
    target += sum(lprior);
    target += std_normal_lpdf(eta);

    target += normal_lpdf(y1 | f[1:N1], sigma);
}


generated quantities {
    array[N] real y;
    y = normal_rng(f, sigma);
}