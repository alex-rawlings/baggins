functions {
    #include helper_funcs.stan
}     


data {
    int<lower=1> N_obs;                  // number of data points
    int<lower=1> N_group;                // number of hierarchical groups
    array[N_obs] int<lower=1, upper=N_group> group_id;  // group id
    vector<lower=0>[N_obs] r;            // radii
    array[N_obs] real<lower=0> density;  // density

    int<lower=1> N_OOS;                  // number of prediction points
    vector<lower=0>[N_OOS] r_OOS;        // prediction radii
    int<lower=1> N_group_OOS;            // prediction groups
    array[N_OOS] int<lower=1> group_id_OOS;  // prediction group ids
}


transformed data {
    array[N_obs] real log10_density = log10(density);
}


parameters {
    // Hyperparameters
    real<lower=-5, upper=10> log10rhoS_mean;
    real<lower=0> log10rhoS_std;
    real<lower=-3, upper=2> log10rS_mean;
    real<lower=0, upper=1> log10rS_std;
    real<lower=-1, upper=2> log10a_mean;
    real<lower=0> log10a_std;
    real<lower=2, upper=8> b_mean;
    real<lower=0> b_std;
    real<lower=0, upper=2.9> g_mean;
    real<lower=0> g_std;

    cholesky_factor_corr[5] L_corr;

    // Non-centered group-level parameters
    matrix[5, N_group] z;

    // Observation noise
    real<lower=0> err;
}

transformed parameters {
    // --- Prior log densities collected for sensitivity analysis ---
    array[11] real lprior;
    lprior[1] = normal_lpdf(log10rhoS_mean | 5, 1);
    lprior[2] = normal_lpdf(log10rhoS_std | 0, 1);
    lprior[3] = normal_lpdf(log10rS_mean | 0, 1);
    lprior[4] = normal_lpdf(log10rS_std | 0, 0.5);
    lprior[5] = normal_lpdf(log10a_mean | 0, 0.5);
    lprior[6] = normal_lpdf(log10a_std | 0, 0.5);
    lprior[7] = normal_lpdf(b_mean | 6, 2);
    lprior[8] = normal_lpdf(b_std | 0, 2);
    lprior[9] = normal_lpdf(g_mean | 0, 1);
    lprior[10] = normal_lpdf(g_std | 0, 1);
    lprior[11] = normal_lpdf(err | 0, 1);

    matrix[5, N_group] theta;
    matrix[5,5] L = diag_pre_multiply([log10rhoS_std, log10rS_std, log10a_std, b_std, g_std], L_corr);

    // group-level parameters
    theta = rep_matrix([log10rhoS_mean, log10rS_mean, log10a_mean, b_mean, g_mean]', N_group) + L * z;

    // protect against extreme draws
    for(i in 1:N_group){
        theta[2, i] = fmin(theta[2, i], 3);     // log10rs < 3
        theta[3, i] = fmax(theta[3, i], -1.0);  // log10a >= -1
        theta[3, i] = fmin(theta[3, i], 2.0);   // log10a < 2
        theta[4, i] = fmax(theta[4, i], 2.0);   // b > 2
        theta[5, i] = fmin(theta[5, i], 2.99);  // g < 3
    }
}

model {
    // Priors (contribute to target)
    target += sum(lprior);
    target += lkj_corr_cholesky_lpdf(L_corr | 3.0);
    to_vector(z) ~ normal(0, 1);

    // Likelihood
    target += reduce_sum(partial_sum_hierarchy, log10_density, 1, r, theta[1]', theta[2]', theta[3]', theta[4]', theta[5]', err, group_id);
}

generated quantities {
    // --- In-sample group-level parameters as vectors for clarity ---
    vector[N_group] log10rhoS  = theta[1]';   // transpose row -> column
    vector[N_group] log10rS = theta[2]';
    vector[N_group] log10a  = theta[3]';
    vector[N_group] b  = theta[4]';
    vector[N_group] g  = theta[5]';

    // transformed parameter not used in sampling
    vector[N_group] rS = pow(10., log10rS);
    vector[N_group] a = pow(10., log10a);

    cov_matrix[5] Sigma = multiply_lower_tri_self_transpose(L);
    corr_matrix[5] Omega = tcrossprod(L_corr);

    // --- declare posterior predictive sets ---
    vector[N_obs] log10_density_posterior;   // posterior predictive draw (with noise)
    vector[N_obs] log10_rho_mean;
    vector[N_obs] density_posterior;
    vector[N_obs] log_lik;

    // Out-of-sample posterior predictions
    vector[N_group_OOS] log10rhoS_OOS;
    vector[N_group_OOS] log10rS_OOS;
    vector[N_group_OOS] log10a_OOS;
    vector[N_group_OOS] b_OOS;
    vector[N_group_OOS] g_OOS;
    vector[N_group_OOS] rS_OOS;
    vector[N_group_OOS] a_OOS;

    vector[N_OOS] log10_rho_mean_OOS;
    vector[N_OOS] log10_density_OOS;
    vector[N_OOS] density_OOS;

    // --- Posterior predictive for observed data ---
    log10_rho_mean = abg_density_vec(r, log10rhoS[group_id], log10rS[group_id], log10a[group_id], b[group_id], g[group_id]);
    log10_density_posterior = to_vector(normal_rng(log10_rho_mean[1:N_obs], err));
    density_posterior = pow(10., log10_density_posterior);
    for(i in 1:N_obs){
        log_lik[i] = normal_lpdf(log10_density[i] | log10_rho_mean[i], err);
    }

    // --- Population draws (hyper-level predictive) ---
    array[N_group_OOS] vector[5] theta_pop;
    for (s in 1:N_group_OOS) {
        theta_pop[s] = multi_normal_cholesky_rng([log10rhoS_mean, log10rS_mean, log10a_mean, b_mean, g_mean]', L);
        log10rhoS_OOS[s] = theta_pop[s][1];
        log10rS_OOS[s] = theta_pop[s][2];
        log10a_OOS[s] = fmax(theta_pop[s][3], -1.0);
        b_OOS[s] = theta_pop[s][4];
        g_OOS[s] = fmin(theta_pop[s][5], 2.99);
    }
    rS_OOS = pow(10., log10rS_OOS);
    a_OOS = pow(10., log10a_OOS);

    log10_rho_mean_OOS = abg_density_vec(
        r_OOS,
        log10rhoS_OOS[group_id_OOS],
        log10rS_OOS[group_id_OOS],
        log10a_OOS[group_id_OOS],
        b_OOS[group_id_OOS],
        g_OOS[group_id_OOS]
    );

    // Guard before exponentiation — inf * anything = inf in density_OOS
    for (i in 1:N_OOS) {
        if (is_inf(log10_rho_mean_OOS[i]) || is_nan(log10_rho_mean_OOS[i])) {
            log10_rho_mean_OOS[i] = not_a_number();
        }
    }

    log10_density_OOS = to_vector(normal_rng(log10_rho_mean_OOS, err));

    density_OOS = pow(10., log10_density_OOS);
}

