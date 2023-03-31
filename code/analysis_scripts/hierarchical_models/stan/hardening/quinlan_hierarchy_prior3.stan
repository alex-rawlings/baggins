functions {
    #include quinlan_funcs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // number of groups
    int<lower=1> N_groups;
    // points per group
    array[N_groups] int<lower=1> N_per_group;
    // group ids
    array[N_tot] int<lower=1> group_id;
    // time values
    array[N_tot] real<lower=0> t;
    // initial eccentricity of galaxy merger
    real<lower=0, upper=1> e_ini;
}


transformed data {
    array[N_groups+1] int<lower=1> new_group_idx;
    new_group_idx[1] = 1;
    for(i in 2:N_groups){
        new_group_idx[i] = new_group_idx[i-1] + N_per_group[i-1];
    }
    new_group_idx[N_groups+1] = new_group_idx[N_groups]+1;
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

    // K parameter
    array[N_groups] real K;
    real K_mean;
    real K_std;

    // y intercept (initial e)
    array[N_groups] real e0;
    real e0_mean;
    real e0_var;

    // general scatter
    real a_err;
    real e_err;

    // forward folded values
    array[N_tot] real inv_a_prior;
    array[N_tot] real e_prior;


    /****** sample parameters ******/
    // slope
    HGp_s_mean = normal_rng(0, 0.3);
    while(HGp_s_mean < 0){
        HGp_s_mean = normal_rng(0, 0.3);
    }
    HGp_s_std = normal_rng(0, 0.3);
    while(HGp_s_std < 0){
        HGp_s_std = normal_rng(0, 0.3);
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

    // K parameter
    K_mean = normal_rng(0, 0.2);
    K_std = normal_rng(0, 0.05);
    while(K_std < 0){
        K_std = normal_rng(0, 0.05);
    }

    // initial eccentricity
    e0_mean = beta_proportion_rng(e_ini, 100);
    //while(e0_mean >= 1){
    //    e0_mean = beta_proportion_rng(e_ini, 100);
    //}
    e0_var = normal_rng(0, 5e3);
    while(e0_var < 0){
        e0_var = normal_rng(0, 5e3);
    }


    // inva error
    a_err = normal_rng(0, 20);
    while(a_err < 0){
        a_err = normal_rng(0, 20);
    }

    // e error
    e_err = normal_rng(0, 100);
    while(e_err < 0){
        e_err = normal_rng(0, 100);
    }

    // sample latent parameters
    for(i in 1:N_groups){
        HGp_s[i] = normal_rng(HGp_s_mean, HGp_s_std);
        while(HGp_s[i] < 0){
            HGp_s[i] = normal_rng(HGp_s_mean, HGp_s_std);
        }

        inv_a_0[i] = normal_rng(inv_a_0_mean, inv_a_0_std);
        while(inv_a_0[i] < 0){
            inv_a_0[i] = normal_rng(inv_a_0_mean, inv_a_0_std);
        }

        K[i] = normal_rng(K_mean, K_std);

        e0[i] = beta_proportion_rng(e0_mean, e0_var);
        while(e0[i] >= 1){
            e0[i] = beta_proportion_rng(e0_mean, e0_var);
        }
    }

    // push forward distributions
    for(i in 1:N_tot){
        {
            // cache means to save computation
            real inv_a_mean_temp = quinlan_inva(t[i], HGp_s[group_id[i]], inv_a_0[group_id[i]]);
            real e_mean_temp = quinlan_e(inv_a_mean_temp, K[group_id[i]], inv_a_0[group_id[i]], e0[group_id[i]]);

            inv_a_prior[i] = normal_rng(inv_a_mean_temp, a_err);
            while(inv_a_prior[i] < 0){
                inv_a_prior[i] = normal_rng(inv_a_mean_temp, a_err);
            }
            //real e_mean_temp = quinlan_e(inv_a_mean_temp, K[group_id[i]], inv_a_0[group_id[i]], e0[group_id[i]]);
            
            e_prior[i] = beta_proportion_rng(e_mean_temp, e_err);
            //while(e_prior[i] >= 1){
            //    e_prior[i] = beta_proportion_rng(e_mean_temp, e_err);
            //}
        }
    }
}