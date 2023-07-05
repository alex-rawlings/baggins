functions {
    #include quinlan_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // number of groups
    int<lower=1> N_groups;
    // group ids
    array[N_tot] int<lower=1> group_id;
    // time values
    array[N_tot] real<lower=0> t;
    // initial eccentricity of galaxy merger
    real<lower=0, upper=1> e_ini;
}



generated quantities {
    // hyperparameters
    real HGp_s_mean = lower_trunc_norm_rng(0, 100, 0);
    real HGp_s_std = lower_trunc_norm_rng(0, 20, 0);
    real inv_a_0_mean = lower_trunc_norm_rng(0, 500, 0);
    real inv_a_0_std = lower_trunc_norm_rng(0, 10, 0);
    real K_mean = normal_rng(0, 0.1);
    real K_std = lower_trunc_norm_rng(0, 0.1, 0);
    real e0_mean = trunc_norm_rng(e_ini, 0.2, 0, 1);
    real e0_std = lower_trunc_norm_rng(0, 0.1, 0);
    real a_err = lower_trunc_norm_rng(0, 0.2, 0);
    real e_err = lower_trunc_norm_rng(0, 2, 0);

    // latent parameters
    array[N_groups] real HGp_s;
    array[N_groups] real inv_a_0;
    array[N_groups] real K;
    array[N_groups] real e0;

    for(i in 1:N_groups){
        HGp_s[i] = lower_trunc_norm_rng(HGp_s_mean, HGp_s_std, 0);
        inv_a_0[i] = lower_trunc_norm_rng(inv_a_0_mean, inv_a_0_std, 0);
        K[i] = normal_rng(K_mean, K_std);
        e0[i] = trunc_norm_rng(e0_mean, e0_std, 0, 1);
    }

    // generate data replication
    array[N_tot] real inv_a_prior;
    array[N_tot] real ecc_prior;

    for(i in 1:N_tot){
            inv_a_prior[i] = lower_trunc_norm_rng(quinlan_inva(t[i], HGp_s[group_id[i]], inv_a_0[group_id[i]]), a_err, 0);
            ecc_prior[i] = trunc_norm_rng(quinlan_e(inv_a_prior[i], K[group_id[i]], inv_a_0[group_id[i]], e0[group_id[i]]), e_err, 0, 1);
    }

}