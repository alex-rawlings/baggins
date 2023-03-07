functions {
    #include quinlan_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // time values
    array[N_tot] real<lower=0> t;
    // inverse semimajor axis
    array[N_tot] real inv_a;
}


parameters {
    real HGp_s;
    real<lower=0> inv_a_0;
    real<lower=0> err;
}


transformed parameters {
    array[N_tot] real<lower=0> inv_a_calc;
    for(i in 1:N_tot){
        inv_a_calc[i] = quinlan_inva(t[i], HGp_s, inv_a_0);
    }
}


model {
    // density at model parameters
    target += normal_lpdf(HGp_s | 0, 0.1);
    target += normal_lpdf(inv_a_0 | inv_a[1], 10);

    target += normal_lpdf(err | 0, 1);

    // likelihood
    target += normal_lpdf(inv_a | inv_a_calc, err);
}


generated quantities {
    array[N_tot] real inv_a_posterior;

    for(i in 1:N_tot){
         inv_a_posterior[i] = normal_rng(quinlan_inva(t[i], HGp_s, inv_a_0), err);
         while(inv_a_posterior[i] < 0){
            inv_a_posterior[i] = normal_rng(quinlan_inva(t[i], HGp_s, inv_a_0), err);
         }
    }
   
   // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] =  normal_lpdf(inv_a | inv_a_calc, err);
    }
}