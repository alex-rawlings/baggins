functions {
    #include binary_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // initial eccentricity
    real<lower=0, upper=1> e_0;
    // normalised angular momentum
    array[N_tot] real log10_angmom;
}


parameters {
    // semimajor axis
    real<lower=0> a_hard;
    // eccentricity
    real<lower=0, upper=1> e_hard;

    // error
    real<lower=0> err;

}

transformed parameters {
    real log10_angmom_calc;
    log10_angmom_calc = binary_log10_angmom(a_hard, e_hard);
}


model {
    // density at model parameters
    target += normal_lpdf(a_hard | 0, 100);
    target += normal_lpdf(e_hard | e_0, 0.3);

    target += normal_lpdf(err | 0, 5);

    // likelihood
    target += normal_lpdf(log10_angmom | log10_angmom_calc, err);
}


generated quantities {
    real log10_angmom_posterior;

    log10_angmom_posterior = normal_rng(binary_log10_angmom(a_hard, e_hard), err);

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_angmom[i] | log10_angmom_calc, err);
    }
}