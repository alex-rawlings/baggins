functions {
    #include binary_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // number of groups
    int<lower=1> N_groups;
    // which group each observation belongs to
    array[N_tot] int<lower=1> group_id;
    // initial eccentricity
    real<lower=0, upper=1> e_0;
    // normalised angular momentum
    array[N_tot] real log10_angmom;
}


parameters {
    // hyperparameters
    real<lower=0> a_hard_mu;
    real<lower=0> a_hard_sigma;
    real<lower=0, upper=1> e_hard_mu;
    real<lower=0> e_hard_sigma;

    // model variance same for all
    real<lower=0> err;

    // latent parameters
    array[N_groups] real<lower=0> a_hard;
    array[N_groups] real<lower=0, upper=1> e_hard;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[4] real lprior;
    lprior[1] = normal_lpdf(a_hard_mu | 0, 150);
    lprior[2] = normal_lpdf(a_hard_sigma | 0, 10);
    lprior[3] = normal_lpdf(e_hard_mu | e_0, 0.3);
    lprior[4] = normal_lpdf(e_hard_sigma | 0, 0.3);

    // deterministic quantity
    array[N_tot] real log10_angmom_calc;
    for(i in 1:N_tot){
        log10_angmom_calc[i] = binary_log10_angmom(a_hard[group_id[i]], e_hard[group_id[i]]);
    }

}


model {
    // density at hyperparameters
    target += sum(lprior);

    // density at error
    target += normal_lpdf(err | 0, 0.5);

    // connect to latent parameters
    target += normal_lpdf(a_hard | a_hard_mu, a_hard_sigma);
    target += normal_lpdf(e_hard | e_hard_mu, e_hard_sigma);

    // likelihood
    target += normal_lpdf(log10_angmom | log10_angmom_calc, err);
}


generated quantities {
    // posterior parameters
    array[N_groups] real a_hard_posterior;
    array[N_groups] real e_hard_posterior;

    // posterior predictive check
    array[N_tot] real log10_angmom_posterior;

    // derived posterior parameters
    // use rejection sampling to constrain to correct domains
    for(i in 1:N_groups){
        a_hard_posterior[i] = normal_rng(a_hard_mu, a_hard_sigma);
        while(a_hard_posterior[i] < 0){
            a_hard_posterior[i] = normal_rng(a_hard_mu, a_hard_sigma);
        }
        e_hard_posterior[i] = normal_rng(e_hard_mu, e_hard_sigma);
        while(e_hard_posterior[i] < 0){
            e_hard_posterior[i] = normal_rng(e_hard_mu, e_hard_sigma);
        }
    }

    // sample posterior
    for(i in 1:N_tot){
        log10_angmom_posterior[i] = binary_log10_angmom(a_hard_posterior[group_id[i]], e_hard_posterior[group_id[i]]);
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_angmom[i] | log10_angmom_calc[i], err);
    }
}