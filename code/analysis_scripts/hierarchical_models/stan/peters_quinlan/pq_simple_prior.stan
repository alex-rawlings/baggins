functions {
    #include pq_funcs.stan
}


data {
    int<lower=1> N_tot;
    real<lower=0> M1;
    real<lower=0> M2;
    real<lower=0> a0;
    real<lower=0, upper=1> e0;
    array[N_tot] real<lower=0> t;
}

transformed data {
    vector[2] ae0;
    ae0[1] = a0;
    ae0[2] = e0;
}


generated quantities {
    real Hps;
    real K;
    real a_err;
    real e_err;
    array[N_tot-1] vector[2] ae;
    array[N_tot] real a_prior;
    array[N_tot] real e_prior;

    // semimajor axis hardening
    Hps = normal_rng(0, 1e2);
    if(Hps < 0){
        Hps = normal_rng(0, 1e2);
    }

    // eccentricity evolution
    K = normal_rng(0, 10);

    // error in semimajor axis
    a_err = normal_rng(0, 10);
    while(a_err < 0){
        a_err = normal_rng(0, 10);
    }

    // error in eccentricity
    e_err = normal_rng(0, 0.1);
    while(e_err < 0){
        e_err = normal_rng(0, 0.1);
    }

    // solve ODE
    ae = ode_rk45(pq_ode, ae0, t[1], t[2:], Hps, K, M1, M2);

    // sample over orbital quantities
    a_prior[1] = normal_rng(a0, a_err);
    while(a_prior[1] < 0){
        a_prior[1] = normal_rng(a0, a_err);
    }
    e_prior[1] = normal_rng(e0, e_err);
    while(e_prior[1] < 0 || e_prior[1] > 1){
        e_prior[1] = normal_rng(e0, e_err);
    }

    for(i in 2:N_tot){
        //a_prior[i] = ae[i-1,1];
        //e_prior[i] = ae[i-1,2];
        a_prior[i] = normal_rng(ae[i-1,1], a_err);
        while(a_prior[i] < 0){
            a_prior[i] = normal_rng(ae[i-1,1], a_err);
        }
        e_prior[i] = normal_rng(ae[i-1,2], e_err);
        while(e_prior[i] < 0 || e_prior[i] > 1){
            e_prior[i] = normal_rng(ae[i-1,2], e_err);
        }
    }
}