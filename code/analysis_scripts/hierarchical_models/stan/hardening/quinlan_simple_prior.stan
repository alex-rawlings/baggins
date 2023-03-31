functions {
    #include quinlan_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // time values
    array[N_tot] real<lower=0> t;
}


generated quantities {
    array[N_tot] real inv_a_prior;
    real HGp_s;
    real inv_a_0;
    real err;

    // sample latent parameters
    HGp_s = normal_rng(0, 0.1);
    while(HGp_s < 0){
        HGp_s = normal_rng(0, 0.1);
    }

    inv_a_0 = normal_rng(0, 10);
    while(inv_a_0 < 0){
        inv_a_0 = normal_rng(0, 10);
    }

    err = normal_rng(0, 1);
    while(err < 0){
        err = normal_rng(0, 1);
    }

    for(i in 1:N_tot){
         inv_a_prior[i] = normal_rng(quinlan_inva(t[i], HGp_s, inv_a_0), err);
         while(inv_a_prior[i] < 0){
            inv_a_prior[i] = normal_rng(quinlan_inva(t[i], HGp_s, inv_a_0), err);
         }
    }
}