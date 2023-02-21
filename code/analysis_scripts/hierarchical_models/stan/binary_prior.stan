data {
    // number of groups
    int<lower=1> N_groups;
}

generated quantities {
    /****** parameters of the parent distributions ******/
    // semimajor axis at time binary is hard
    array[N_groups] real a_hard;
    real a_hard_mu;
    real a_hard_sigma;

    // eccentricity at time binary is hard
    array[N_groups] real e_hard;
    real e_hard_a;
    real e_hard_b;

    // general scatter
    real err;

    /****** forward folded values ******/
    array[N_groups] real log_angmom;

    /****** sample the parameters ******/
    // semimajor axis
    a_hard_mu = normal_rng(0, 20);
    while(a_hard_mu < 0){
        a_hard_mu = normal_rng(0, 20);
    }
    a_hard_sigma = normal_rng(0, 10);
    while(a_hard_sigma < 0){
        a_hard_sigma = normal_rng(0, 10);
    }

    // eccentricity
    e_hard_a = normal_rng(0, 10);
    while(e_hard_a < 0 || e_hard_a > 1){
        e_hard_a = normal_rng(0, 10);
    }
    e_hard_b = normal_rng(0, 10);
    while(e_hard_b < 0){
        e_hard_b = normal_rng(0, 10);
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
        e_hard[i] = beta_rng(e_hard_a, e_hard_b);

        log_angmom[i] = normal_rng(0.5 * (log10(a_hard[i] * (1.0 - pow(e_hard[i], 2.0)))), err);
    }
}