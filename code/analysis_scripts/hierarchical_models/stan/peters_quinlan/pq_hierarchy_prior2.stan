functions {
    #include pq_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // number of groups
    int<lower=1> N_groups;
    // points per group
    array[N_groups] int<lower=1> points_per_group;
    // time values
    array[N_tot] real<lower=0> tIn;
    // initial semimajor axis
    array[N_groups] real<lower=0> a0;
    // initial eccentricity
    array[N_groups] real<lower=0, upper=1> e0;
    // masses
    real<lower=0> M1;
    real<lower=0> M2;
}


transformed data {
    array[N_groups] vector[2] ae0;
    for(i in 1:N_groups){
        ae0[i][1] = a0[i];
        ae0[i][2] = e0[i];
    }
}


generated quantities {
    // hyper parameters 
    real Hps_mean;
    real Hps_std;
    real K_mean;
    real K_std;
    real a_err;
    real e_err;

    // latent parameters
    array[N_groups] real Hps;
    array[N_groups] real K;

    // determined values
    array[N_tot-N_groups] vector[2] ae;
    array[N_tot] real a_prior;
    array[N_tot] real e_prior;

    // semimajor axis hardening
    Hps_mean = normal_rng(0, 1e2);
    while(Hps_mean < 0){
        Hps_mean = normal_rng(0, 1e2);
    }
    Hps_std = normal_rng(0, 10);
    while(Hps_std < 0){
        Hps_std = normal_rng(0, 10);
    }

    // eccentricity evolution
    K_mean = normal_rng(0, 1);
    K_std = normal_rng(0, 0.1);
    while(K_std < 0){
        K_std = normal_rng(0, 0.1);
    }

    // quantity errors
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

    // sample latent parameters
    {
        int start_idx = 1;
        int end_idx;
        for(i in 1:N_groups){
            Hps[i] = normal_rng(Hps_mean, Hps_std);
            while(Hps[i] < 0){
                Hps[i] = normal_rng(Hps_mean, Hps_std);
            }
            K[i] = normal_rng(K_mean, K_std);
            
            end_idx = start_idx + points_per_group[i];
            ae[start_idx:end_idx] = ode_rk45(pq_ode, ae0[i], t[start_idx], t[start_idx+1:end_idx], Hps[i], K[i], M1, M2);
            start_idx = end_idx + 1;
            
            a_prior[start_idx] = normal_rng(a0[i], a_err);
            while(a_prior[start_idx < 0]){
                a_prior[start_idx] = normal_rng(a0[i], a_err);
            }
            e_prior[start_idx] = normal_rng(a0[i], e_err);
            while(e_prior[start_idx < 0]){
                e_prior[start_idx] = normal_rng(e0[i], e_err);
            }

            for(j in start_idx+1:end_idx){
                a_prior[j] = normal_rng(ae[j-1,1], a_err);
                while(a_prior[j] < 0){
                    a_prior[j] = normal_rng(ae[j-1,1], a_err);
                }
                e_prior[j] = normal_rng(ae[j-1,2], e_err);
                while(e_prior[j] < 0){
                    e_prior[j] = normal_rng(ae[j-1,2], e_err);
                }
            }

        }
    }

}

