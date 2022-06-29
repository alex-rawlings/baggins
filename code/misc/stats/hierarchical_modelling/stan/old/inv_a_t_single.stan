data {
    int<lower=1> N_child;
    int<lower=0> N_tot;
    array[N_tot] int<lower=1, upper=N_child> child_id;
    vector[N_tot] t;
    vector[N_tot] a;
}

transformed data {
    //vector[N_tot] inv_a = 1 ./ a;
    vector[N_tot] inv_a_recentre = 1 ./ (a - mean(a));
    vector[N_tot] t_recentre = t - mean(t);
}

parameters {
    real<lower= 0> sigma;

    // parameters for each child
    real HGp_s;
    real c;
}

model {
    // priors
    sigma ~ cauchy(0, 0.1);

    HGp_s ~ normal(0, 0.1);
    c ~ normal(0, 100);

    inv_a_recentre ~ normal(HGp_s * t_recentre + c, sigma);
}