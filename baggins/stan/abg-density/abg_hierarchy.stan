functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N;                  // number of data points
    vector[N] r;                     // radii
    array[N] real<lower=0> density;               // observed log10(density)

    // Individual groups
    int<lower=1> N_groups;   // number of groups
    array[N] int<lower=1, upper=N_groups> group_idx; // indexing of observations to group

    // OOS inputs
    int<lower=0> N_OOS;                           // number of prediction points
    vector<lower=0, upper=max(r)>[N_OOS] r_OOS;   // radii at which to predict
    int<lower=1> N_groups_OOS;                    // total number of individual OOS groups
    array[N_OOS] int<lower=1> group_idx_OOS;      // group ids for generated quantities
}

transformed data {
    array[N] real log10_density = log10(density);
    real median_r = quantile(r, 0.5);
    int N_GQ = N + N_OOS;
    int N_groups_GQ = N_groups + N_groups_OOS;
    vector<lower=0, upper=max(r)>[N_GQ] r_GQ = append_row(r, r_OOS);
    array[N_GQ] int<lower=1> group_idx_GQ = append_array(group_idx, group_idx_OOS);
}


parameters {
    // hyperparameters
    real<lower=-5, upper=10> log10rhoS_mean;
    real<lower=0> log10rhoS_std;
    real log10rS_mean;
    real<lower=0> log10rS_std;
    real a_mean;
    real<lower=0> a_std;
    real b_mean;
    real<lower=0> b_std;
    real g_mean;
    real<lower=0> g_std;
    real<lower=0> err0;
    real err_grad;

    // define latent parameters for each group
    vector<lower=-5, upper=15>[N_groups] log10rhoS;
    vector<lower=-5, upper=2>[N_groups] log10rS;
    vector<lower=-10, upper=10>[N_groups] a;
    vector<lower=-10, upper=10>[N_groups] b;
    vector<lower=-3, upper=3>[N_groups] g;
}


transformed parameters {
    array[12] real lprior;
    lprior[1] = normal_lpdf(log10rhoS_mean | 5, 1);
    lprior[2] = normal_lpdf(log10rhoS_std | 0, 1);
    lprior[3] = normal_lpdf(log10rS_mean | 0, 1);
    lprior[4] = normal_lpdf(log10rS_std | 0, 0.5);
    lprior[5] = normal_lpdf(a_mean | 0, 4);
    lprior[6] = normal_lpdf(a_std | 0, 2);
    lprior[7] = normal_lpdf(b_mean | 0, 4);
    lprior[8] = normal_lpdf(b_std | 0, 2);
    lprior[9] = normal_lpdf(g_mean | 0, 2);
    lprior[10] = normal_lpdf(g_std | 0, 2);
    lprior[11] = normal_lpdf(err0 | 0, 1);
    lprior[12] = normal_lpdf(err_grad | 0, 1);

    // define error
    vector[N] err = radially_vary_err(r, err0, err_grad, median_r);
}


model {
    target += sum(lprior);
    // connect to latent parameters
    target += normal_lpdf(log10rhoS | log10rhoS_mean, log10rhoS_std);
    target += normal_lpdf(log10rS | log10rS_mean, log10rS_std);
    target += normal_lpdf(a | a_mean, a_std);
    target += normal_lpdf(b | b_mean, b_std);
    target += normal_lpdf(g | g_mean, g_std);

    // likelihood
    target += reduce_sum(partial_sum_hierarchy, log10_density, 1, r, log10rhoS, log10rS, a, b, g, err, group_idx);
}


generated quantities {
    // transformed parameter not used in sampling
    vector[N_groups] rS = pow(10., log10rS);

    vector[N_GQ] log10_rho_posterior;   // posterior predictive draw (with noise)
    vector[N_GQ] log10_rho_mean;
    vector[N_GQ] rho_posterior;
    vector[N_GQ] err_posterior = radially_vary_err(r_GQ, err0, err_grad, median_r);

    // posterior parameters
    vector[N_groups_GQ] log10rhoS_posterior;
    vector[N_groups_GQ] log10rS_posterior;
    vector[N_groups_GQ] a_posterior;
    vector[N_groups_GQ] b_posterior;
    vector[N_groups_GQ] g_posterior;
    vector[N_groups_GQ] rS_posterior;

    for(i in 1:N_groups_GQ){
        log10rhoS_posterior[i] = normal_rng(log10rhoS_mean, log10rhoS_std);
        log10rS_posterior[i] = normal_rng(log10rS_mean, log10rS_std);
        a_posterior[i] = normal_rng(a_mean, a_std);
        b_posterior[i] = normal_rng(b_mean, b_std);
        g_posterior[i] = normal_rng(g_mean, g_std);
    }
    rS_posterior = pow(10., log10rS_posterior);

    // push forward data
    log10_rho_mean = abg_density_vec(r_GQ, log10rhoS_posterior[group_idx_GQ], log10rS_posterior[group_idx_GQ], a_posterior[group_idx_GQ], b_posterior[group_idx_GQ], g_posterior[group_idx_GQ]);
    for(i in 1:N_GQ){
        log10_rho_posterior[i] = trunc_normal_rng(log10_rho_mean[i], err_posterior[i], -2, 15);
    }
    rho_posterior = pow(10., log10_rho_posterior);
}