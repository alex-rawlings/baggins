data {
    int<lower=1> N_child;
    int<lower=0> N_tot;
    array[N_tot] int<lower=1, upper=N_child> child_id;
    vector[N_tot] t;
    vector[N_tot] inv_a;
    vector[N_tot] inv_a_err;
}

transformed data {
    vector[N_tot] inv_a_recentre = inv_a;// - mean(inv_a);
    vector[N_tot] t_recentre = t - mean(t);
}

parameters {
    // parameters of the parent distributions
    real HGp_s_mu;
    real<lower=0> HGp_s_tau;
    real c_mu;
    real<lower=0> c_tau;

    // parameters for each child
    vector[N_child] HGp_s;
    vector[N_child] c;
}

transformed parameters {
    vector[N_tot] inv_a_true = HGp_s[child_id] .* t_recentre + c[child_id];
}

model {
    // priors
    HGp_s_mu ~ normal(0, 10);
    HGp_s_tau ~ cauchy(0, 5);
    c_mu ~ normal(0, 100);
    c_tau ~cauchy(0, 5);

    HGp_s ~ normal(HGp_s_mu, HGp_s_tau);
    c ~ normal(c_mu, c_tau);

    //HGp_s ~ normal(0, 10);
    //c ~ normal(0, 100);

    inv_a_recentre ~ normal(inv_a_true, inv_a_err);
}