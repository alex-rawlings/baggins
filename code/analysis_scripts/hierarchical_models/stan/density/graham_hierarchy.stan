functions{
    #include funcs_graham.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // number of groups
    int<lower=1> N_groups;
    // number of observations per group
    array[N_groups] int<lower=1> N_per_group;
    // array of radial values
    array[N_tot] real<lower=0.0> R;
    // array of surface density values
    array[N_tot] real<lower=0.0> log10_surf_rho;
}


parameters {
    // hierarchy: observations belong to different groups
    // introduce hyperparameters
    real<lower=0.0> r_b_mean;
    real<lower=0.0> r_b_std;
    real<lower=0.0> Re_mean;
    real<lower=0.0> Re_std;
    real<lower=0.0> log10_I_b_mean;
    real<lower=0.0> log10_I_b_std;
    real<lower=0.0> g_mean;
    real<lower=0.0> g_std;
    real<lower=0.0, upper=20.0> n_mean;
    real<lower=0.0> n_std;
    real<lower=0.0> a_mean;
    real<lower=0.0> a_std;

    // model variance same for all
    real<lower=0.0> err;

    // define latent parameters
    array[N_groups] real<lower=0.0> r_b;
    array[N_groups] real<lower=0.0> Re;
    array[N_groups] real<lower=0.0> log10_I_b;
    array[N_groups] real<lower=0.0> g;
    array[N_groups] real<lower=0.0> n;
    array[N_groups] real<lower=0.0> a;
}


transformed parameters {
    // prior information for sensitivity analysis
    array[12] real lprior;
    lprior[1] = normal_lpdf(r_b_mean | 0.0, 2.0);
    lprior[2] = normal_lpdf(r_b_std | 0.0, 1.0);
    lprior[3] = normal_lpdf(Re_mean | 0.0, 14.0);
    lprior[4] = normal_lpdf(Re_std | 0.0, 6.0);
    lprior[5] = normal_lpdf(log10_I_b_mean | 0.0, 20.0);
    lprior[6] = normal_lpdf(log10_I_b_std | 0.0, 10.0);
    lprior[7] = normal_lpdf(g_mean | 0.0, 0.5);
    lprior[8] = normal_lpdf(g_std | 0.0, 1.0);
    lprior[9] = normal_lpdf(n_mean | 0.0, 8.0);
    lprior[10] = normal_lpdf(n_std | 0.0, 5.0);
    lprior[11] = normal_lpdf(a_mean | 0.0, 20.0);
    lprior[12] = normal_lpdf(a_std | 0.0, 10.0);

    // deterministic surface density calculation
    array[N_tot] real log10_surf_rho_calc;

    {
        int start_idx = 1;
        int end_idx;
        for(i in 1:N_groups){
            end_idx = start_idx + N_per_group[i] - 1;
            log10_surf_rho_calc[start_idx:end_idx] = log10_I(N_per_group[i], R[start_idx:end_idx], log10_I_b[i], g[i], a[i], r_b[i], Re[i], n[i]);
            start_idx = end_idx + 1;
        }
    }
}


model {
    // density at hyperparameters
    target += sum(lprior);

    // density at error
    target += normal_lpdf(err | 0.0, 1.0);

    // connect to latent parameters
    target += normal_lpdf(r_b | r_b_mean, r_b_std);
    target += normal_lpdf(Re | Re_mean, Re_std);
    target += normal_lpdf(log10_I_b | log10_I_b_mean, log10_I_b_std);
    target += normal_lpdf(g | g_mean, g_std);
    target += normal_lpdf(n | n_mean, n_std);
    target += normal_lpdf(a | a_mean, a_std);

    // likelihood
    target += normal_lpdf(log10_surf_rho | log10_surf_rho_calc, err);
}


generated quantities {
    // posterior parameters
    real r_b_posterior = normal_rng(r_b_mean, r_b_std);
    real Re_posterior = normal_rng(Re_mean, Re_std);
    real log10_I_b_posterior = normal_rng(log10_I_b_mean, log10_I_b_std);
    real g_posterior = normal_rng(g_mean, g_std);
    real n_posterior = normal_rng(n_mean, n_std);
    real a_posterior = normal_rng(a_mean, a_std);

    // posterior predictive check
    // note we need to use rejection sampling here
    array[N_tot] real log10_surf_rho_posterior;
    array[1] real Ri;
    array[1] real Ii;
    for(i in 1:N_tot){
        Ri[1] = R[i];
        Ii = normal_rng(log10_I(1, Ri, log10_I_b_posterior, g_posterior, a_posterior, r_b_posterior, Re_posterior, n_posterior), err);
        while(log10_surf_rho_posterior[i] < 0.0){
            Ii = normal_rng(log10_I(1, Ri, log10_I_b_posterior, g_posterior, a_posterior, r_b_posterior, Re_posterior, n_posterior), err);
        }
        log10_surf_rho_posterior[i] = Ii[1];
    }

    /****** determine log likelihood function ******/
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_calc[i], err);
    }

}