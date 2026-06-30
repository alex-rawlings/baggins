functions{
    #include funcs_graham.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N_obs;                  // number of data points
    int<lower=1> N_group;                // number of hierarchical groups
    array[N_obs] int<lower=1, upper=N_group> group_id;  // group id
    vector<lower=0>[N_obs] R;            // radii
    array[N_obs] real<lower=0> density;  // density

    int<lower=1> N_OOS;                  // number of prediction points
    vector<lower=0>[N_OOS] R_OOS;        // prediction radii
    int<lower=1> N_group_OOS;            // prediction groups
    array[N_OOS] int<lower=1> group_id_OOS;  // prediction group ids
}


transformed data {
    array[N_obs] real log10_density = log10(density);
}



parameters {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real log10densb_mean;
    real<lower=0> log10densb_std;
    real<upper=2> log10rb_mean;
    real<lower=0> log10rb_std;
    real<upper=2> log10g_mean;
    real<lower=0> log10g_std;
    real<upper=1> log10n_mean;
    real<lower=0> log10n_std;
    real<upper=2> log10a_mean;
    real<lower=0> log10a_std;
    real<upper=2> log10Re_mean;
    real<lower=0> log10Re_std;

    cholesky_factor_corr[6] L_corr;

    // Non-centered group-level parameters
    matrix[6, N_group] z;

    // Observation noise
    real<lower=0> err;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[13] real lprior;
    lprior[1] = normal_lpdf(log10densb_mean | 10, 2);
    lprior[2] = normal_lpdf(log10densb_std | 0, 1);
    lprior[3] = normal_lpdf(log10rb_mean | -1, 1);
    lprior[4] = normal_lpdf(log10rb_std | 0, 0.5);
    lprior[5] = normal_lpdf(log10g_mean | -1, 1);
    lprior[6] = normal_lpdf(log10g_std | 0, 0.5);
    lprior[7] = normal_lpdf(log10n_mean | 0, 0.5);
    lprior[8] = normal_lpdf(log10n_std | 0, 0.5);
    lprior[9] = normal_lpdf(log10a_mean | 0, 1);
    lprior[10] = normal_lpdf(log10a_std | 0, 1);
    lprior[11] = normal_lpdf(log10Re_mean | 1, 1);
    lprior[12] = normal_lpdf(log10Re_std | 0, 1);
    lprior[13] = normal_lpdf(err | 0, 1);

    matrix[6, N_group] theta;
    matrix[6, 6] L = diag_pre_multiply([log10densb_std, log10rb_std, log10g_std, log10n_std, log10a_std, log10Re_std], L_corr);

    // group-level parameters
    theta = rep_matrix([log10densb_mean, log10rb_mean, log10g_mean, log10n_mean, log10a_mean, log10Re_mean]', N_group) + L * z;

    // TODO protect against extreme draws?
    /*for(i in 1:N_group){
        theta[2, i] = fmin(theta[2, i], 3);     // log10rs < 3
    }*/
}


model {
    // density at hyperparameters
    target += sum(lprior);
    target += lkj_corr_cholesky_lpdf(L_corr | 3.0);
    to_vector(z) ~ normal(0, 1);

    // likelihood
    // TODO update parameters here
    target += reduce_sum(partial_sum_hierarchy, log10_density, 1, R, theta[1]', theta[2]', theta[3]', theta[4]', theta[5]', theta[6]', err, group_id);
}


generated quantities {
    // --- In-sample group-level parameters as vectors for clarity ---
    vector[N_group] log10densb = theta[1]';  // transpose row -> column
    vector[N_group] lgo10rb = theta[2]';
    vector[N_group] log10g = theta[3]';
    vector[N_group] log10n = theta[4]';
    vector[N_group] log10a = theta[5]';
    vector[N_group] log10Re = theta[6]';

    // transformed parameters not used in sampling
    vector[N_group] rb = pow(10., log10rb);
    vector[N_group] g = pow(10., log10g);
    vector[N_group] n = pow(10., log10n);
    vector[N_group] a = pow(10., log10a);
    vector[N_group] Re = pow(10., log10Re);

    // --- declare posterior predictive sets ---
    vector[N_obs] log10_density_posterior;   // posterior predictive draw (with noise)
    vector[N_obs] log10_Sigma_mean;
    vector[N_obs] density_posterior;
    vector[N_obs] log_lik;

    // Out-of-sample posterior predictions
    vector[N_group_OOS] log10densb_OOS;
    vector[N_group_OOS] log10rb_OOS;
    vector[N_group_OOS] log10g_OOS;
    vector[N_group_OOS] log10n_OOS;
    vector[N_group_OOS] log10a_OOS;
    vector[N_group_OOS] log10Re_OOS;
    vector[N_group_OOS] rb_OOS;
    vector[N_group_OOS] g_OOS;
    vector[N_group_OOS] n_OOS;
    vector[N_group_OOS] a_OOS;
    vector[N_group_OOS] Re_OOS;

    vector[N_OOS] log10_Sigma_mean_OOS;
    vector[N_OOS] log10_density_OOS;
    vector[N_OOS] density_OOS;

    // --- Posterior predictive for observed data ---
    log10_Sigma_mean = graham_surf_density_vec(
                            R,
                            log10densb[group_id],
                            lgo10rb[group_id],
                            log10g[group_id],
                            log10n[group_id],
                            log10a[group_id],
                            log10Re[group_id]
                        );
    log10_density_posterior = to_vector(normal_rng(log10_Sigma_mean[1:N_obs], err));
    density_posterior = pow(10., log10_density_posterior);
    for(i in 1:N_obs){
        log_lik[i] = normal_lpdf(log10_density[i] | log10_Sigma_mean[i], err);
    }

    // --- Population draws (hyper-level predictive) ---
    array[N_group_OOS] vector[6] theta_pop;
    for (s in 1:N_group_OOS) {
        theta_pop[s] = multi_normal_cholesky_rng([log10densb_mean, log10rb_mean, log10g_mean, log10n_mean, log10a_mean, log10Re_mean]', L);
        log10densb_OOS[s] = theta_pop[s][1];
        log10rb_OOS[s] = theta_pop[s][2];
        log10g_OOS[s] = theta_pop[s][3];
        log10n_OOS[s] = theta_pop[s][4];
        log10a_OOS[s] = theta_pop[s][5];
        log10Re_OOS[s] = theta_pop[s][6];
    }
    rb_OOS = pow(10., log10rb_OOS);
    g_OOS = pow(10., log10g_OOS);
    n_OOS = pow(10., log10n_OOS);
    a_OOS = pow(10., log10a_OOS);
    Re_OOS = pow(10., log10Re_OOS);

    log10_Sigma_mean_OOS = graham_surf_density_vec(
                            R_OOS,
                            log10densb_OOS[group_id_OOS],
                            lgo10rb_OOS[group_id_OOS],
                            log10g_OOS[group_id_OOS],
                            log10n_OOS[group_id_OOS],
                            log10a_OOS[group_id_OOS],
                            log10Re_OOS[group_id_OOS]
                        );

    // Guard before exponentiation — inf * anything = inf in density_OOS
    for (i in 1:N_OOS) {
        if (is_inf(log10_Sigma_mean_OOS[i]) || is_nan(log10_Sigma_mean_OOS[i])) {
            log10_Sigma_mean_OOS[i] = not_a_number();
        }
    }

    log10_density_OOS = to_vector(normal_rng(log10_Sigma_mean_OOS, err));

    density_OOS = pow(10., log10_density_OOS);
}
