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
    // inverse semimajor axis values
    array[N_tot] real<lower=0> inv_a;
}


transformed data {
    array[N_tot] real log_t = log(t);
}


parameters {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real<lower=0> HGp_s_mean;
    real<lower=0> HGp_s_std;
    real<lower=0> inv_a_0_mean;
    real<lower=0> inv_a_0_std;

    // model variance same for all
    real<lower=0> err;

    // define latent parameters
    array[N_tot] real<lower=0> HGp_s;
    array[N_tot] real<lower=0> inv_a_0;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[5] real lprior;
    lprior[1] = normal_lpdf(HGp_s_mean | 0, 0.3);
    lprior[2] = normal_lpdf(HGp_s_std | 0, 0.3);
    lprior[3] = normal_lpdf(inv_a_0_mean | 0, 20);
    lprior[4] = normal_lpdf(inv_a_0_std | 0, 5);
    lprior[5] = normal_lpdf(err | 0, 0.2);

    // deterministic hardening rate calculation
    array[N_tot] real inv_a_calc;

    {
        real tempval;
        for(i in 1:N_tot){
            tempval = log_quinlan_inva(log_t[i], HGp_s[group_id[i]]);
            inv_a_calc[i] = exp(tempval) + inv_a_0[group_id[i]];
        }
    }
}


model {
    // density at hyperparameters
    target += sum(lprior);

    //connect to latent parameters
    target += normal_lpdf(HGp_s | HGp_s_mean, HGp_s_std);
    target += normal_lpdf(inv_a_0 | inv_a_0_mean, inv_a_0_std);

    // likelihood
    target += normal_lpdf(inv_a | inv_a_calc, err);
}


generated quantities {
    array[N_groups] real HGp_s_posterior;
    array[N_groups] real inv_a_0_posterior;

    // posterior predictive check
    array[N_tot] real inv_a_posterior;

    // use rejection sampling to constrain to positive values
    for(i in 1:N_groups){
        HGp_s_posterior[i] = normal_rng(HGp_s_mean, HGp_s_std);
        while(HGp_s_posterior[i] < 0){
            HGp_s_posterior[i] = normal_rng(HGp_s_mean, HGp_s_std);
        }
        inv_a_0_posterior[i] = normal_rng(inv_a_0_mean, inv_a_0_std);
        while(inv_a_0_posterior[i] < 0){
            inv_a_0_posterior[i] = normal_rng(inv_a_0_mean, inv_a_0_std);
        }
    }

    for(i in 1:N_tot){
        inv_a_posterior[i] = normal_rng(quinlan_inva(t[i], HGp_s_posterior[group_id[i]], inv_a_0_posterior[group_id[i]]), err);
        while(inv_a_posterior[i] < 0){
            inv_a_posterior[i] = normal_rng(quinlan_inva(t[i], HGp_s_posterior[group_id[i]], inv_a_0_posterior[group_id[i]]), err);
        }
    }

    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(inv_a[i] | inv_a_calc[i], err);
    }

}