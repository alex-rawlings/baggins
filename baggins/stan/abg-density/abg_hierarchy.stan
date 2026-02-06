functions {
    #include helper_funcs.stan
    #include ../custom_rngs.stan
}


data {
    int<lower=1> N_obs;
    int<lower=1> N_group;
    array[N_obs] int<lower=1, upper=N_group> group_id;
    vector<lower=0>[N_obs] r;
    array[N_obs] real<lower=0> density;

    int<lower=1> N_OOS;
    vector<lower=0>[N_OOS] r_OOS;
    int<lower=1> N_group_OOS;
    array[N_OOS] int<lower=1> group_id_OOS;      // group ids for generated quantities
}


transformed data {
    array[N_obs] real log10_density = log10(density);
    int N_GQ = N_obs + N_OOS;
    int N_group_GQ = N_group + N_group_OOS;
    vector<lower=0, upper=max(r)>[N_GQ] r_GQ = append_row(r, r_OOS);
    array[N_GQ] int<lower=1> group_id_GQ = append_array(group_id, group_id_OOS);
    
}


parameters {
    // Hyperparameters
    real<lower=-5, upper=10> log10rhoS_mean;
    real<lower=0> log10rhoS_std;
    real log10rS_mean;
    real<lower=0> log10rS_std;
    real log10a_mean;
    real<lower=0> log10a_std;
    real b_mean;
    real<lower=0> b_std;
    real g_mean;
    real<lower=0> g_std;

    cholesky_factor_corr[5] L_corr;

    // Non-centered group-level parameters
    matrix[5, N_group] z;

    // Observation noise
    real<lower=0> obs_sigma;
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
    lprior[7] = normal_lpdf(b_mean | 0, 4);
    lprior[8] = normal_lpdf(b_std | 0, 2);
    lprior[9] = normal_lpdf(g_mean | 0, 2);
    lprior[10] = normal_lpdf(g_std | 0, 2);
    lprior[11] = normal_lpdf(obs_sigma | 0, 1);

    matrix[5, N_group] theta;
    matrix[5,5] L = diag_pre_multiply([log10rhoS_std, log10rS_std, log10a_std, b_std, g_std], L_corr);

    // group-level parameters
    theta = rep_matrix([log10rhoS_mean, log10rS_mean, log10a_mean, b_mean, g_mean]', N_group) + L * z;
}

model {
    // Priors (contribute to target)
    target += sum(lprior);
    L_corr ~ lkj_corr_cholesky(2.0);
    to_vector(z) ~ normal(0, 1);

    // Likelihood
    target += reduce_sum(partial_sum_hierarchy, log10_density, 1, r, theta[1]', theta[2]', theta[3]', theta[4]', theta[5]', obs_sigma, group_id);
}

generated quantities {
    // --- Group-level parameters as vectors for clarity ---
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

    // --- declare posterior predictive and OOS sets ---
    vector[N_GQ] log10_rho_posterior;   // posterior predictive draw (with noise)
    vector[N_GQ] log10_rho_mean;
    vector[N_GQ] rho_posterior;

    // posterior parameters
    vector[N_group_OOS] log10rhoS_posterior;
    vector[N_group_OOS] log10rS_posterior;
    vector[N_group_OOS] log10a_posterior;
    vector[N_group_OOS] b_posterior;
    vector[N_group_OOS] g_posterior;
    vector[N_group_OOS] rS_posterior;
    vector[N_group_OOS] a_posterior;

    // --- Posterior predictive for observed data ---
    log10_rho_mean[1:N_obs] = abg_density_vec(r, log10rhoS[group_id], log10rS[group_id], log10a[group_id], b[group_id], g[group_id]);
    log10_rho_posterior[1:N_obs] = to_vector(normal_rng(log10_rho_mean[1:N_obs], obs_sigma));

    // --- Population draws (hyper-level predictive) ---
    array[N_group_OOS] vector[5] theta_pop;
    for (s in 1:N_group_OOS) {
        theta_pop[s] = multi_normal_cholesky_rng([log10rhoS_mean, log10rS_mean, log10a_mean, b_mean, g_mean]', L);
        log10rhoS_posterior[s] = theta_pop[s][1];
        log10rS_posterior[s] = theta_pop[s][2];
        log10a_posterior[s] = theta_pop[s][3];
        b_posterior[s] = theta_pop[s][4];
        g_posterior[s] = theta_pop[s][5];
    }
    rS_posterior = pow(10., log10rS_posterior);
    a_posterior = pow(10., log10a_posterior);

    log10_rho_mean[N_obs+1:N_GQ] = abg_density_vec(
        r_OOS,
        log10rhoS_posterior[group_id_OOS],
        log10rS_posterior[group_id_OOS],
        log10a_posterior[group_id_OOS],
        b_posterior[group_id_OOS],
        g_posterior[group_id_OOS]
    );
    log10_rho_posterior[N_obs+1:N_GQ] = to_vector(normal_rng(log10_rho_mean[N_obs+1:N_GQ], obs_sigma));

    rho_posterior = pow(10., log10_rho_posterior);
}

