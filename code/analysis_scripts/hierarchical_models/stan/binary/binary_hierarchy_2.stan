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
    // reduced orbital energy
    array[N_tot] real log10_energy;
}


parameters {
    // hyperparameters
    real<lower=0> a_hard_mu;
    real<lower=0> a_hard_sigma;
    real<lower=0, upper=1> e_hard_mu;
    real<lower=0> e_hard_sigma;

    // model variance same for all
    real<lower=0> L_err;
    real<lower=0> E_err;

    // latent parameters
    array[N_groups] real<lower=0> a_hard;
    array[N_groups] real<lower=0, upper=1> e_hard;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[6] real lprior;
    lprior[1] = normal_lpdf(a_hard_mu | 0, 150);
    lprior[2] = normal_lpdf(a_hard_sigma | 0, 10);
    lprior[3] = normal_lpdf(e_hard_mu | e_0, 0.3);
    lprior[4] = normal_lpdf(e_hard_sigma | 0, 0.3);
    lprior[5] =  normal_lpdf(L_err | 0, 0.5);
    lprior[6] =  normal_lpdf(E_err | 0, 0.5);

    // deterministic quantity
    array[N_tot] real log10_energy_calc;
    array[N_tot] real log10_angmom_calc;
    for(i in 1:N_tot){
        log10_energy_calc[i] = binary_log10_energy(a_hard[group_id[i]]);
        log10_angmom_calc[i] = binary_log10_angmom(a_hard[group_id[i]], e_hard[group_id[i]]);
    }
}


model {
    // density at hyperparameters
    target += sum(lprior);

    // connect to latent parameters
    target += normal_lpdf(a_hard | a_hard_mu, a_hard_sigma);
    target += normal_lpdf(e_hard | e_hard_mu, e_hard_sigma);

    // likelihood
    target += normal_lpdf(log10_energy | log10_energy_calc, E_err);
    target += normal_lpdf(log10_angmom | log10_angmom_calc, L_err);
}


generated quantities {
    // sample posterior for predictive check
    array[N_groups] real log10_energy_posterior_mean;
    array[N_groups] real log10_angmom_posterior_mean;
    array[N_tot] real log10_energy_posterior;
    array[N_tot] real log10_angmom_posterior;

    for(i in 1:N_groups){
        log10_energy_posterior_mean[i] = binary_log10_energy(a_hard[i]);
        log10_angmom_posterior_mean[i] = binary_log10_angmom(a_hard[i], e_hard[i]);
    }

    for(i in 1:N_tot){
        log10_energy_posterior[i] = normal_rng(log10_energy_posterior_mean[group_id[i]], E_err);
        log10_angmom_posterior[i] = normal_rng(log10_angmom_posterior_mean[group_id[i]], L_err);
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_energy[i] | log10_energy_calc[i], E_err) + normal_lpdf(log10_angmom[i] | log10_angmom_calc[i], L_err);
    }
}