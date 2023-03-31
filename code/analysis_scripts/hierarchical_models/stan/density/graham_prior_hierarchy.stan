functions {
    #include funcs_graham.stan
}


data {
    int<lower=1> N_tot;
    int<lower=1> N_groups;
    array[N_tot] int<lower=1> group_id;
    array[N_tot] real<lower=0.0> R;
}


generated quantities {
    // hyperpriors
    real log10_r_b_mean;
    real r_b_std;
    real log10_Re_mean;
    real Re_std;
    real log10_I_b_mean;
    real log10_I_b_std;
    real log10_g_mean;
    real g_std;
    real log10_n_mean;
    real n_std;
    real log10_a_mean;
    real a_std;

    // transformed latent parameters
    array[N_groups] real r_b;
    array[N_groups] real Re;
    array[N_groups] real log10_I_b;
    array[N_groups] real g;
    array[N_groups] real n;
    array[N_groups] real a;

    // relation error
    real err = normal_rng(0, 1);
    while(err < 0){
        err = normal_rng(0, 1);
    }

    // prior check
    array[N_tot] real log10_surf_rho_prior;

    // sample hyperpriors
    log10_r_b_mean = normal_rng(0.0, 0.7);
    r_b_std = normal_rng(0, 1);
    while(r_b_std<0){
        r_b_std = normal_rng(0, 1);
    }
    log10_Re_mean = normal_rng(0.85, 0.5);
    Re_std = normal_rng(0, 4);
    while(Re_std<0){
        Re_std = normal_rng(0, 4);
    }
    log10_I_b_mean = normal_rng(0, 3);
    log10_I_b_std = normal_rng(0, 5);
    while(log10_I_b_std < 0){
        log10_I_b_std = normal_rng(0, 5);
    }
    log10_g_mean = normal_rng(-0.5, 0.2);
    g_std = normal_rng(0, 1);
    while(g_std<0){
        g_std = normal_rng(0, 1);
    }
    log10_n_mean = normal_rng(0.6, 0.4);
    while(log10_n_mean>1.3){
        log10_n_mean = normal_rng(0.6, 0.4);
    }
    n_std = normal_rng(0, 2);
    while(n_std<0) {
        n_std = normal_rng(0, 5);
    }
    log10_a_mean = normal_rng(1.0, 0.5);
    a_std = normal_rng(0, 10);
    while(a_std<0){
        a_std = normal_rng(0, 10);
    }

    // sample latent parameters and prior check
    {
        array[N_groups] real b_param;
        array[N_groups] real pre_term;

        for(i in 1:N_groups){
            r_b[i] = normal_rng(pow(10,log10_r_b_mean), r_b_std);
            while(r_b[i] < 0){
                r_b[i] = normal_rng(pow(10,log10_r_b_mean), r_b_std);
            }
            Re[i] = normal_rng(pow(10,log10_Re_mean), Re_std);
            while(Re[i] < 0){
                Re[i] = normal_rng(pow(10,log10_Re_mean), Re_std);
            }
            log10_I_b[i] = normal_rng(log10_I_b_mean, log10_I_b_std);
            g[i] = normal_rng(pow(10,log10_g_mean), g_std);
            while(g[i] < 0){
                g[i] = normal_rng(pow(10,log10_g_mean), g_std);
            }
            n[i] = normal_rng(pow(10,log10_n_mean), n_std);
            while(n[i] < 0){
                n[i] = normal_rng(pow(10,log10_n_mean), n_std);
            }
            a[i] = normal_rng(pow(10,log10_a_mean), a_std);
            while(a[i] < 0){
                a[i] = normal_rng(pow(10,log10_a_mean), a_std);
            }
            b_param[i] = sersic_b_parameter(n[i]);
            pre_term[i] = graham_preterm(g[i], a[i], n[i], b_param[i], r_b[i], Re[i]);
        }

        for(i in 1:N_tot){
            log10_surf_rho_prior[i] = normal_rng(graham_surf_density(R[i], pre_term[group_id[i]], g[group_id[i]], a[group_id[i]], r_b[group_id[i]], n[group_id[i]], b_param[group_id[i]], Re[group_id[i]], log10_I_b[group_id[i]]), err);
        }
    }
}