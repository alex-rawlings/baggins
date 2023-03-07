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
}


generated quantities {
    /****** parameters of the parent distributions ******/
    // slope
    array[N_groups] real HGp_s;
    real HGp_s_mean;
    real HGp_s_std;

    // y intercept (initial 1/a)
    array[N_groups] real inv_a_0;
    real inv_a_0_mean;
    real inv_a_0_std;

    // general scatter
    real err;

    // forward folded values
    array[N_tot] real inv_a_prior;


    /****** sample parameters ******/
    // slope
    HGp_s_mean = normal_rng(0, 0.1);
    while(HGp_s_mean < 0){
        HGp_s_mean = normal_rng(0, 0.1);
    }
    HGp_s_std = normal_rng(0, 0.1);
    while(HGp_s_std < 0){
        HGp_s_std = normal_rng(0, 0.1);
    }

    // intercept
    inv_a_0_mean = normal_rng(0, 20);
    while(inv_a_0_mean < 0){
        inv_a_0_mean = normal_rng(0, 20);
    }
    inv_a_0_std = normal_rng(0, 5);
    while(inv_a_0_std < 0){
        inv_a_0_std = normal_rng(0, 5);
    }

    // error
    err = normal_rng(0, 0.2);
    while(err < 0){
        err = normal_rng(0, 0.2);
    }

    for(i in 1:N_groups){
        HGp_s[i] = normal_rng(HGp_s_mean, HGp_s_std);
        while(HGp_s[i] < 0){
            HGp_s[i] = normal_rng(HGp_s_mean, HGp_s_std);
        }

        inv_a_0[i] = normal_rng(inv_a_0_mean, inv_a_0_std);
        while(inv_a_0[i] < 0){
            inv_a_0[i] = normal_rng(inv_a_0_mean, inv_a_0_std);
        }
    }

    for(i in 1:N_tot){
        inv_a_prior[i] = normal_rng(quinlan_inva(t[i], HGp_s[group_id[i]], inv_a_0[group_id[i]]), err);
        while(inv_a_prior[i] < 0){
            inv_a_prior[i] = normal_rng(quinlan_inva(t[i], HGp_s[group_id[i]], inv_a_0[group_id[i]]), err);
        }
    }
}