functions {
    #include helpers.stan
}


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
    vector[N_tot] t_transformed = (t - t_mean) / t_sigma;
}


generated quantities {
    /**** forward declaration of variables ****/
    // variables for d(1/a)/dt relation
    real inv_a_mean;
    real HGp_s_mu_scaled_prior;
    real<lower=0> HGp_s_tau_scaled_prior;
    real c_mu_prior;
    real<lower=0> c_tau_prior;
    real<lower=0> sigma_inv_a_prior;

    vector[N_child] HGp_s_scaled_prior;
    vector[N_child] HGp_s_prior;
    vector[N_child] c_prior;
    vector[N_tot] inv_a_recentre_prior;
    vector[N_tot] inv_a_prior;
    vector[N_tot] a_prior;

    // variables for de/dt relation
    real K_mu_prior;
    real K_tau_prior;
    real<lower=0> sigma_e_prior;

    vector[N_child] K_prior;
    vector[N_tot] del_e_prior;
    
    // helper variables for the integral{a} calculation
    array[N_tot] real temp_integrated_a_;
    array[2] real temp_theta;
    array[0] real temp_x_r;
    array[0] int temp_x_i;

    // sample the prior distributions
    // d(1/a)/dt relation
    HGp_s_mu_scaled_prior = normal_rng(0, 1);
    HGp_s_tau_scaled_prior = cauchy_rng(0, 1);
    c_mu_prior = normal_rng(0, 1);
    c_tau_prior = cauchy_rng(0, 1);
    sigma_inv_a_prior = cauchy_rng(0, 1);

    // de/dt relation
    K_mu_prior = normal_rng(0, 1);
    K_tau_prior = cauchy_rng(0, 1);
    sigma_e_prior = cauchy_rng(0, 1);

    for(n in 1:N_child){
        HGp_s_scaled_prior[n] = normal_rng(HGp_s_mu_scaled_prior, HGp_s_tau_scaled_prior);
        HGp_s_prior[n] = HGp_s_scaled_prior[n] / t_sigma;
        c_prior[n] = normal_rng(c_mu_prior, c_tau_prior);
        K_prior[n] = normal_rng(K_mu_prior, K_tau_prior);
    }

    for(n in 1:N_tot){
        // sample prior distribution of inverse a and a
        inv_a_recentre_prior[n] = normal_rng(HGp_s_scaled_prior[child_id[n]] .* t_transformed[n] + c_prior[child_id[n]], sigma_inv_a_prior);
    }
    inv_a_mean = mean(inv_a_recentre_prior);

    for (n in 1:N_tot) {
        inv_a_prior[n] = inv_a_recentre_prior[n] + inv_a_mean;
        a_prior[n] = 1 ./ inv_a_prior[n];

        // integrate sampled value of a
        temp_theta = {HGp_s_prior[child_id[n]], c_prior[child_id[n]]+t_mean};
        temp_integrated_a_[n] = integrate_1d(a_integrand, t[1], t[n], temp_theta, temp_x_r, temp_x_i);

        // sample delta e
        del_e_prior[n] = normal_rng(K_prior[child_id[n]] .* HGp_s_scaled_prior[child_id[n]] .* temp_integrated_a_[n], sigma_e_prior);
    }
}