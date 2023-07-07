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
    // inverse semimajor axis values
    array[N_tot] real<lower=0> inv_a;
    // eccentricity values
    array[N_tot] real<lower=0, upper=1> ecc;
    // total number of out-of-sample points
    int<lower=1> N_OOS;
    // group ids for generated quantities
    array[N_OOS] int<lower=1> group_id_OOS;
    // out-of-sample time values
    array[N_OOS] real<lower=0, upper=max(t)> t_OOS;
}


transformed data {
    int N_GQ = N_tot + N_OOS;
}


parameters {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real<lower=0> HGp_s_mean;
    real<lower=0> HGp_s_std;
    real<lower=0> inv_a_0_mean;
    real<lower=0> inv_a_0_std;
    real K_mean;
    real<lower=0> K_std;
    real<lower=0, upper=1> e0_mean;
    real<lower=0> e0_std;

    // model variance same for all
    // error in semimajor axis
    real<lower=0> a_err;
    // error in eccentricity
    real<lower=0> e_err;

    // define latent parameters
    array[N_groups] real<lower=0> HGp_s;
    array[N_groups] real<lower=0> inv_a_0;
    array[N_groups] real K;
    array[N_groups] real<lower=0, upper=1> e0;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[10] real lprior;
    lprior[1] = normal_lpdf(HGp_s_mean | 0, 100);
    lprior[2] = normal_lpdf(HGp_s_std | 0, 20);
    lprior[3] = normal_lpdf(inv_a_0_mean | 0, 500);
    lprior[4] = normal_lpdf(inv_a_0_std | 0, 10);
    lprior[5] = normal_lpdf(K_mean | 0, 0.1);
    lprior[6] = normal_lpdf(K_std | 0, 0.1);
    lprior[7] = normal_lpdf(e0_mean | e_ini, 0.2);
    lprior[8] = normal_lpdf(e0_std | 0, 0.1);
    lprior[9] = normal_lpdf(a_err | 0, 0.2);
    lprior[10] = normal_lpdf(e_err | 0, 2);

    // deterministic hardening rate calculation
    array[N_tot] real<lower=0> inv_a_calc;
    array[N_tot] real e_calc;

    for(i in 1:N_tot){
        inv_a_calc[i] = quinlan_inva(t[i], HGp_s[group_id[i]], inv_a_0[group_id[i]]);
        e_calc[i] = quinlan_e(inv_a_calc[i], K[group_id[i]], inv_a_0[group_id[i]], e0[group_id[i]]);
    }
}


model {
    // density at hyperparameters
    target += sum(lprior);

    //connect to latent parameters
    target += normal_lpdf(HGp_s | HGp_s_mean, HGp_s_std);
    target += normal_lpdf(inv_a_0 | inv_a_0_mean, inv_a_0_std);
    target += normal_lpdf(K | K_mean, K_std);
    target += normal_lpdf(e0 | e0_mean, e0_std);

    // likelihood
    target += normal_lpdf(inv_a | inv_a_calc, a_err);
    target += normal_lpdf(ecc | e_calc, e_err);
}


generated quantities {
    array[N_groups] real HGp_s_posterior;
    array[N_groups] real inv_a_0_posterior;
    array[N_groups] real K_posterior;
    array[N_groups] real e0_posterior;

    // generate data replication
    array[N_GQ] real t_GQ = append_array(t, t_OOS);
    array[N_GQ] int group_id_GQ = append_array(group_id, group_id_OOS);
    array[N_GQ] real inv_a_posterior;
    array[N_GQ] real ecc_posterior;

    for(i in 1:N_groups){
        HGp_s_posterior[i] = lower_trunc_norm_rng(HGp_s_mean, HGp_s_std, 0.0);
        inv_a_0_posterior[i] = lower_trunc_norm_rng(inv_a_0_mean, inv_a_0_std, 0.0);
        K_posterior[i] = normal_rng(K_mean, K_std);
        e0_posterior[i] = trunc_norm_rng(e0_mean, e0_std, 0.0, 1.0);
    }

    for(i in 1:N_GQ){
            inv_a_posterior[i] = lower_trunc_norm_rng(quinlan_inva(t_GQ[i], HGp_s_posterior[group_id_GQ[i]], inv_a_0_posterior[group_id_GQ[i]]), a_err, 0.0);

            ecc_posterior[i] = trunc_norm_rng(quinlan_e(inv_a_posterior[i], K_posterior[group_id_GQ[i]], inv_a_0_posterior[group_id_GQ[i]], e0_posterior[group_id_GQ[i]]), e_err, 0.0, 1.0);
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(inv_a[i] | inv_a_calc[i], a_err) + normal_lpdf(ecc[i] | e_calc[i], e_err);
    }

}