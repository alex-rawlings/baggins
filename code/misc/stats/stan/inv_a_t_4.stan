data {
    // number of simulation children
    int<lower=1> N_child;
    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // id of the child simulation that each point belongs to
    array[N_tot] int<lower=1, upper=N_child> child_id;
    // vector of time values
    vector[N_tot] t;
    // vector of semimajor axis values
    vector[N_tot] a;
}

transformed data {
    // for reparameterisation
    real t_mean = mean(t);
    real t_sigma = sd(t);

    // rescale and recentre data
    vector[N_tot] inv_a = 1 ./ a;
    vector[N_tot] inv_a_recentre = inv_a - mean(inv_a);
    vector[N_tot] t_transformed = (t - mean(t)) / t_sigma;

}

parameters {
    // parameters of the parent distributions
    real HGp_s_mu_scaled;
    real<lower=0> HGp_s_tau_scaled;
    real c_mu;
    real<lower=0> c_tau;
    real<lower= 0> sigma;

    // parameters for each child
    vector[N_child] HGp_s_scaled;
    vector[N_child] c;
}

transformed parameters {
    vector[N_tot] inv_a_true;
    vector[N_child] HGp_s;
    // unscaled value
    real HGp_s_mu;
    real<lower=0> HGp_s_tau;

    // connection between 1/a and t
    inv_a_true = HGp_s_scaled[child_id] .* t_transformed + c[child_id];

    // determine unscaled quantities
    HGp_s[child_id] = HGp_s_scaled[child_id] / t_sigma;
    HGp_s_mu = HGp_s_mu_scaled / t_sigma;
    HGp_s_tau = HGp_s_tau_scaled / t_sigma;
}

model {
    // priors
    HGp_s_mu_scaled ~ normal(0, 1);
    HGp_s_tau_scaled ~ cauchy(0, 5);
    c_mu ~ normal(0, 1);
    c_tau ~ cauchy(0, 5);
    sigma ~ cauchy(0, 1);

    // connection to latent parameters
    HGp_s_scaled ~ normal(HGp_s_mu_scaled, HGp_s_tau_scaled);
    c ~ normal(c_mu, c_tau);

    // likelihood
    inv_a_recentre ~ normal(inv_a_true, sigma);
}