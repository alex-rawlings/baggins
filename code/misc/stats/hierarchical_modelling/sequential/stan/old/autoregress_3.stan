functions {
    real get_scaled_inv_chi2_mode(real nu, real tau){
        return nu * tau^2 / (nu + 2);
    }
}


data{
    int<lower=0> N_tot;
    vector[N_tot] t;
    vector[N_tot] inv_a;
    vector[N_tot] inv_a_err;
}

transformed data {
    vector[N_tot] inv_a_recentre = inv_a - mean(inv_a);
}

parameters{
    real alpha_mu;
    real<lower=0> alpha_tau;
    real beta_mu;
    real<lower=0> beta_tau;

    real alpha;
    real beta;

    real HGp_s;
    real c;
    real<lower=0> sigma2_reg;
    real<lower=0> sigma2_H;
}

transformed parameters {
    real<lower=0> sigma_reg = sqrt(sigma2_reg);
    real<lower=0> sigma_H = sqrt(sigma2_H);
}

model {
    // priors
    alpha_mu ~ normal(0, 1000);
    alpha_tau ~ cauchy(0, 5);
    beta_mu ~ normal(0, 1000);
    beta_tau ~ cauchy(0, 5);

    //sigma2_reg ~ cauchy(0, 5);
    //sigma2_H ~ cauchy(0, 5);

    // connection to regression parameters
    alpha ~ normal(alpha_mu, alpha_tau);
    beta ~ normal(beta_mu, beta_tau);

    inv_a_recentre[2:N_tot] ~ normal(beta + alpha * inv_a_recentre[1:(N_tot-1)], sigma_reg);
    HGp_s ~ normal((inv_a_recentre-c)./t, sigma_H);
}