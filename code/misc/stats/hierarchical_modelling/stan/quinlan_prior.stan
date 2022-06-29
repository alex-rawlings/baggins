data{
    // number of simulation children
    int<lower=1> N_child;
    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // id of the child simulation that each point belongs to
    array[N_tot] int<lower=1, upper=N_child> child_id;
    // vector of time values
    vector[N_tot] t;
}

transformed data {
    // for reparameterisation
    real t_mean = mean(t);
    real t_sigma = sd(t);

    // rescale and recentre data
    vector[N_tot] t_transformed = (t - mean(t)) / t_sigma;
}

generated quantities {
    // forward declaration of variables
    real HGp_s_mu_scaled_prior;
    real<lower=0> HGp_s_tau_scaled_prior;
    real c_mu_prior;
    real<lower=0> c_tau_prior;
    real<lower=0> sigma_prior;

    vector[N_child] HGp_s_scaled_prior;
    vector[N_child] HGp_s_prior;
    vector[N_child] c_prior;
    vector[N_tot] inv_a_prior;
    vector[N_tot] a_prior;

    // sample the prior distributions
    HGp_s_mu_scaled_prior = normal_rng(0, 1);
    HGp_s_tau_scaled_prior = cauchy_rng(0, 5);
    c_mu_prior = normal_rng(0, 1);
    c_tau_prior = cauchy_rng(0, 5);
    sigma_prior = cauchy_rng(0, 1);

    for(n in 1:N_child){
        HGp_s_scaled_prior[n] = normal_rng(HGp_s_mu_scaled_prior, HGp_s_tau_scaled_prior);
        HGp_s_prior[n] = HGp_s_scaled_prior[n] / t_sigma;
        c_prior[n] = normal_rng(c_mu_prior, c_tau_prior);
    }

    for(n in 1:N_tot){
        inv_a_prior[n] = normal_rng(HGp_s_scaled_prior[child_id[n]] .* t_transformed[n] + c_prior[child_id[n]], sigma_prior);
        a_prior[n] = 1.0 ./ inv_a_prior[n];
    }
}