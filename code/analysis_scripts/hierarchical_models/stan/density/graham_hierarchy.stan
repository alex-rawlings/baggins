functions{
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // array of radial values
    vector<lower=0>[N_tot] R;
    // array of surface density values
    array[N_tot] real<lower=0.0> log10_surf_rho;

    // Individual groups
    // number of groups
    int<lower=1> N_groups;
    // indexing of observations to group
    array[N_tot] int<lower=1, upper=N_groups> group_idx;

    // Out of Sample points
    // follows same structure as above
    // total number of OOS points
    int<lower=1> N_OOS;
    // total number of individual OOS groups
    int<lower=1> N_groups_OOS;
    // OOS radii values
    vector<lower=0, upper=max(R)>[N_OOS] R_OOS;
    // group ids for generated quantities
    array[N_OOS] int<lower=1> group_idx_OOS;

}



transformed data {
    int N_GQ = N_tot + N_OOS;
    int N_groups_GQ = N_groups + N_groups_OOS;
    vector<lower=0, upper=max(R)>[N_GQ] R_GQ = append_row(R, R_OOS);
    array[N_GQ] int<lower=1> group_idx_GQ = append_array(group_idx, group_idx_OOS);
}



parameters {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real log10densb_mean;
    real<lower=0> log10densb_std;
    real<lower=0> g_lam;
    real<lower=0> rb_sig;
    real<lower=0, upper=15> n_mean;
    real<lower=0> n_std;
    real<lower=0> a_sig;
    real<lower=0> Re_sig;
    real<lower=0> err_mean;
    real<lower=0> err_std;

    // model variance, function of radius
    vector<lower=0>[N_tot] err;

    // define latent parameters for each group
    vector<lower=0, upper=5>[N_groups] rb;
    vector<lower=0, upper=20>[N_groups] Re;
    vector<lower=0, upper=20>[N_groups] n;
    vector<lower=0, upper=2>[N_groups] g;
    vector<lower=-5, upper=15>[N_groups] log10densb;
    vector<lower=0, upper=15>[N_groups] a;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[10] real lprior;
    lprior[1] = normal_lpdf(log10densb_mean | 10, 1);
    lprior[2] = normal_lpdf(log10densb_std | 0, 0.05);
    lprior[3] = exponential_lpdf(g_lam | 10);
    lprior[4] = normal_lpdf(rb_sig | 0, 0.2);
    lprior[5] = normal_lpdf(n_mean | 4, 2);
    lprior[6] = normal_lpdf(n_std | 0, 2);
    lprior[7] = gamma_lpdf(a_sig | 2, 0.2);
    lprior[8] = normal_lpdf(Re_sig | 0, 20);
    lprior[9] = normal_lpdf(err_mean | 0, 1);
    lprior[10] = normal_lpdf(err_std | 0, 0.2);
}


model {
    // density at hyperparameters
    target += sum(lprior);

    // density at error
    target += normal_lpdf(err | err_mean, err_std);

    // connect to latent parameters
    target += normal_lpdf(log10densb | log10densb_mean, log10densb_std);
    target += exponential_lpdf(g | g_lam);
    target += rayleigh_lpdf(rb | rb_sig);
    target += normal_lpdf(n | n_mean, n_std);
    target += rayleigh_lpdf(a | a_sig);
    target += rayleigh_lpdf(Re | Re_sig);

    // likelihood
    target += reduce_sum(partial_sum_hierarchy, log10_surf_rho, 1, N_groups, R, g, a, rb, n, Re, log10densb, err, group_idx);
}


generated quantities {
    // posterior parameters
    vector[N_groups_GQ] log10densb_posterior;
    vector[N_groups_GQ] g_posterior;
    vector[N_groups_GQ] rb_posterior;
    vector[N_groups_GQ] n_posterior;
    vector[N_groups_GQ] a_posterior;
    vector[N_groups_GQ] Re_posterior;
    vector[N_GQ] err_posterior;

    // generate data replication
    vector[N_GQ] log10_surf_rho_posterior;
    vector[N_GQ] surf_rho_posterior;


    {
        vector[N_groups_GQ] pre_term;
        vector[N_groups_GQ] b_param;
        for(i in 1:N_groups_GQ){
            log10densb_posterior[i] = trunc_normal_rng(log10densb_mean, log10densb_std, -5, 15);
            g_posterior[i] = trunc_exponential_rng(g_lam, 0, 2);
            rb_posterior[i] = trunc_rayleigh_rng(rb_sig, 0, 5);
            n_posterior[i] = trunc_normal_rng(n_mean, n_std, 0, 20);
            a_posterior[i] = trunc_rayleigh_rng(a_sig, 0, 15);
            Re_posterior[i] = trunc_rayleigh_rng(Re_sig, 0, 20);
        }
        // some helper quantities
        b_param = sersic_b_parameter(n_posterior);
        pre_term = graham_preterm(g_posterior, a_posterior, n_posterior, b_param, rb_posterior, Re_posterior);

        // push forward data
        vector[N_GQ] mean_gsd = graham_surf_density_vec(
                                        R_GQ,
                                        pre_term[group_idx_GQ],
                                        g_posterior[group_idx_GQ],
                                        a_posterior[group_idx_GQ],
                                        rb_posterior[group_idx_GQ],
                                        n_posterior[group_idx_GQ],
                                        b_param[group_idx_GQ],
                                        Re_posterior[group_idx_GQ],
                                        log10densb_posterior[group_idx_GQ]);
        for(i in 1:N_GQ){
            err_posterior[i] = lower_trunc_normal_rng(err_mean, err_std, 0.);
            log10_surf_rho_posterior[i] = trunc_normal_rng(mean_gsd[i], err_posterior[i], -5, 15);
        }
    }
    surf_rho_posterior = pow(10., log10_surf_rho_posterior);

}
