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
}

generated quantities {
    // forward declaration of variables
    real HGp_s_mu_prior;
    real<lower=0> HGp_s_tau_prior;
    real c_mu_prior;
    real<lower=0> c_tau_prior;
    real<lower=0> sigma_prior;

    vector[N_child] HGp_s_prior;
    vector[N_child] c_prior;
    array[N_tot] real inv_a_prior_exp;
    array[N_tot] real inv_a_prior;
    array[N_tot] real a_prior;

    // sample the prior distributions
    HGp_s_mu_prior = normal_rng(0, 1);
    HGp_s_tau_prior = cauchy_rng(0, 5);
    c_mu_prior = normal_rng(0, 1);
    c_tau_prior = cauchy_rng(0, 5);
    sigma_prior = cauchy_rng(0, 1);

    for(n in 1:N_child){
        HGp_s_prior[n] = lognormal_rng(HGp_s_mu_prior, HGp_s_tau_prior);
        c_prior[n] = lognormal_rng(c_mu_prior, c_tau_prior);
    }

    inv_a_prior_exp = lognormal_rng(HGp_s_prior[child_id] .* t + c_prior[child_id], sigma_prior);
    inv_a_prior = log(inv_a_prior_exp);
    a_prior = inv(inv_a_prior);
}