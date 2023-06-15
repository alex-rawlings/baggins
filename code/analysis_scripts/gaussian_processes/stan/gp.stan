// Gaussian process fitting following example "Pedictive inference with a GP" in
// https://mc-stan.org/docs/2_19/stan-users-guide/fit-gp-section.html
data {
    // number of data points to infer on
    int<lower=1> N1;
    // observed independent variable
    array[N1] real<lower=0> theta1;
    // observed dependent variable
    vector<lower=0, upper=1>[N1] ecc;
    // number of data points to predict on
    int<lower=1> N2;
    // independent variable to predict on
    array[N2] real<lower=0> theta2;
}


transformed data {
    real delta = 1e-9;
    int<lower=1> N = N1 + N2;

    array[N] real theta;
    for(i in 1:N1){
        theta[i] = theta1[i];
    }
    for(i in 1:N2){
        theta[N1+i] = theta2[i];
    }

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
    target += inv_gamma_lpdf(rho | 3, 2);
    target += std_normal_lpdf(alpha);
    target += std_normal_lpdf(sigma);
    target += std_normal_lpdf(eta);

    target += normal_lpdf(ecc | f[1:N1], sigma);
}


generated quantities {
    array[N2] real y;
    real nanval = y[1];
    int counter = 1;

    for(i in 1:N2){
        y[i] = normal_rng(f[N1+i], sigma);
        while(y[i] < 0 || y[i] > 1 || counter>1000){
            y[i] = normal_rng(f[N1+i], sigma);
            counter+=1;
        }
        if(counter>1000){
            y[i] = nanval;
        }
    }
}