data {
    int<lower=0> N_tot;
    vector[N_tot] t;
    vector[N_tot] inv_a;
    real<lower=0> t_err;
}

transformed data {
    vector[(N_tot-1)] diff_t = t[2:N_tot] - t[1:(N_tot-1)];
    vector[(N_tot-1)] diff_inv_a = inv_a[2:N_tot] - inv_a[1:(N_tot-1)];

    vector[(N_tot-1)] diff_t_recentre = diff_t - mean(diff_t);
    vector[(N_tot-1)] diff_inv_a_recentre = diff_inv_a - mean(diff_inv_a);
}

parameters {
    real HGp_s_mu;
    real<lower=0> HGp_s_tau;
    real t_mu;
    real<lower=0> t_tau;

    real diff_t_true;
    real HGp_s;
    real<lower=0> sigma;
}

model {
    // priors
    t_mu ~ normal(diff_t_recentre, 100);
    t_tau ~ cauchy(0, 10);
    HGp_s_mu ~ normal(0, 0.1);
    HGp_s_tau ~ cauchy(0, 10);
    sigma ~ cauchy(0, 10);

    diff_t_true ~ normal(t_mu, t_tau);
    HGp_s ~ normal(HGp_s_mu, HGp_s_tau);

    diff_inv_a_recentre ~ normal(HGp_s .* diff_t_true, sigma);
}