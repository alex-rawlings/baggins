functions {
    #include binary_funcs.stan
}

data {
    // number of groups
    int<lower=1> N_groups;
    // initial eccentricity
    real<lower=0, upper=1> e_0;
}

generated quantities {
    /****** parameters of the parent distributions ******/
    // semimajor axis at time binary is hard
    array[N_groups] real a_hard;
    real a_hard_mu_kpc;
    real a_hard_mu;
    real a_hard_sigma;

    // eccentricity at time binary is hard
    array[N_groups] real e_hard;
    real e_hard_mu;
    real e_hard_sigma;

    // general scatter
    real err;

    /****** forward folded values ******/
    array[N_groups] real log_angmom;

    /****** sample the parameters ******/
    // semimajor axis
    a_hard_mu = normal_rng(0, 200);
    while(a_hard_mu < 0){
        a_hard_mu_kpc = normal_rng(0, 200);
    }
    //a_hard_mu = a_hard_mu_kpc * 1000;
    a_hard_sigma = normal_rng(0, 10);
    while(a_hard_sigma < 0){
        a_hard_sigma = normal_rng(0, 10);
    }

    // eccentricity
    e_hard_mu = normal_rng(e_0, 0.3);
    while(e_hard_mu < 0 || e_hard_mu > 1){
        e_hard_mu = normal_rng(e_0, 0.3);
    }
    e_hard_sigma = normal_rng(0, 0.3);
    while(e_hard_sigma < 0){
        e_hard_sigma = normal_rng(0, 0.3);
    }

    err = normal_rng(0, 0.5);
    while(err < 0){
        err = normal_rng(0, 0.5);
    }

    for(i in 1:N_groups){
        a_hard[i] = normal_rng(a_hard_mu, a_hard_sigma);
        while(a_hard[i] < 0){
            a_hard[i] = normal_rng(a_hard_mu, a_hard_sigma);
        }

        e_hard[i] = normal_rng(e_hard_mu, e_hard_sigma);
        while(e_hard[i] < 0 || e_hard[i] > 1){
            e_hard[i] = normal_rng(e_hard_mu, e_hard_sigma);
        }

        log_angmom[i] = normal_rng(binary_log10_angmom(a_hard[i], e_hard[i]), err);
    }
}