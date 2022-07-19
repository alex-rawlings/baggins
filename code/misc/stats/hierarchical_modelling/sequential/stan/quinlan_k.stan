functions {
    #include helpers.stan
}


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
    //vector of delta eccentricity
    vector[N_tot] delta_e;
}


transformed data {
    // for reparameterisation
    real t_mean = mean(t);
    real t_sigma = sd(t);
    real inv_a_mean;

    // rescale and recentre data
    vector[N_tot] inv_a = 1 ./ a;
    inv_a_mean = mean(inv_a);
    vector[N_tot] inv_a_recentre = inv_a - inv_a_mean;
    vector[N_tot] t_transformed = (t - mean(t)) / t_sigma;

    // helper variables for the integral{a} calculation
    array[0] real temp_x_r;
    array[0] int temp_x_i;

}


parameters {
    /**** parameters of the parent distributions ****/
    // parameters of d(1/a)/dt relation
    real HGp_s_mu_scaled;
    real<lower=0> HGp_s_tau_scaled;
    real c_mu;
    real<lower=0> c_tau;
    real<lower= 0> sigma_inv_a;

    // parameters for each child
    vector[N_child] HGp_s_scaled;
    vector[N_child] c;

    // parameters for de/dt relation
    real K_mu;
    real K_tau;
    real<lower=0> sigma_e;

    // parameters for each child
    vector[N_child] K;
}


transformed parameters {
    vector[N_child] HGp_s;
    vector[N_tot] inv_a_true;
    
    HGp_s[child_id] = HGp_s_scaled[child_id] / t_sigma;

    // connection between 1/a and t
    inv_a_true = HGp_s_scaled[child_id] .* t_transformed + c[child_id];


    vector[N_tot] temp_integrated_a_;
    //array[2] real temp_theta;
    temp_integrated_a_[1] = 0.0;
    for(n in 2:N_tot){
        temp_integrated_a_[n] = integrate_1d(a_integrand, t[1], t[n], {HGp_s[child_id[n]], c[child_id[n]]+t_mean}, temp_x_r, temp_x_i);
    }
    
}


model {

    /**** sample prior distributions ****/
    // d(1/a)/dt relation
    HGp_s_mu_scaled ~ normal(0, 1);
    HGp_s_tau_scaled ~ cauchy(0, 1);
    c_mu ~ normal(0, 1);
    c_tau ~ cauchy(0, 1);
    sigma_inv_a ~ cauchy(0, 1);

    // de/dt relation
    K_mu ~ normal(0, 1);
    K_tau ~ cauchy(0, 1);
    sigma_e ~ cauchy(0, 1);

    // connection to latent parameters
    HGp_s_scaled ~ normal(HGp_s_mu_scaled, HGp_s_tau_scaled);
    c ~ normal(c_mu, c_tau);
    K ~ normal(K_mu, K_tau);

    // likelihood
    inv_a_recentre ~ normal(inv_a_true, sigma_inv_a);
   
    delta_e ~ normal(K[child_id] .* HGp_s_scaled[child_id] .* temp_integrated_a_, sigma_e);
}


generated quantities {
    // unscaled value
    real HGp_s_mu;
    real<lower=0> HGp_s_tau;

    // determine unscaled quantities
    HGp_s_mu = HGp_s_mu_scaled / t_sigma;
    HGp_s_tau = HGp_s_tau_scaled / t_sigma;

    // posterior predictive values
    vector[N_tot] inv_a_posterior;
    vector[N_tot] delta_e_posterior;

    // perform posterior predictive check
    inv_a_posterior = normal_rng(HGp_s_mu_scaled, HGp_s_tau_scaled) * t_transformed + normal_rng(c_mu, c_tau) + inv_a_mean;

     delta_e_posterior = normal_rng(K_mu, K_tau) * normal_rng(HGp_s_mu_scaled, HGp_s_tau_scaled) * temp_integrated_a_;

}