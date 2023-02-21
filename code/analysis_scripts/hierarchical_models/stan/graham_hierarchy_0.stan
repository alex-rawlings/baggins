functions{
    #include funcs_graham.stan
}


data {
    // total number of points
    int<lower=1> N_tot;
    // number of groups
    int<lower=1> N_groups;
    // which group each observation belongs to
    array[N_tot] int<lower=1> group_id;
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
        array[N_groups] real pre_term;
        array[N_groups] real b_param;
        for(i in 1:N_groups){
            b_param[i] = sersic_b_parameter(n[i]);
            pre_term[i] = graham_preterm(g[i], a[i], n[i], b_param[i], r_b[i], Re[i]);
        }
        for(i in 1:N_tot){
            log10_surf_rho_calc[i] = graham_surf_density(R[i], pre_term[group_id[i]], g[group_id[i]], a[group_id[i]], r_b[group_id[i]], n[group_id[i]], b_param[group_id[i]], Re[group_id[i]], log10_I_b[group_id[i]]);
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
    array[N_groups] real r_b_posterior;
    array[N_groups] real Re_posterior;
    array[N_groups] real log10_I_b_posterior;
    array[N_groups] real g_posterior;
    array[N_groups] real n_posterior;
    array[N_groups] real a_posterior;

    // posterior predictive check
    array[N_tot] real log10_surf_rho_posterior;

    // derived posterior parameters
    {
        array[N_groups] real b_param_posterior;
        array[N_groups] real pre_term_posterior;

        // use rejection sampling to constrain to >0 values
        for(i in 1:N_groups){
            r_b_posterior[i] = normal_rng(r_b_mean, r_b_std);
            while(r_b_posterior[i] < 0){
                r_b_posterior[i] = normal_rng(r_b_mean, r_b_std);
            }
            Re_posterior[i] = normal_rng(Re_mean, Re_std);
            while(Re_posterior[i] < 0){
                Re_posterior[i] = normal_rng(Re_mean, Re_std);
            }
            log10_I_b_posterior[i] = normal_rng(log10_I_b_mean, log10_I_b_std);
                while(log10_I_b_posterior[i] < 0){
                    log10_I_b_posterior[i] = normal_rng(log10_I_b_mean, log10_I_b_std);
                }
            g_posterior[i] = normal_rng(g_mean, g_std);
            while(g_posterior[i] < 0){
                g_posterior[i] = normal_rng(g_mean, g_std);
            }
            n_posterior[i] = normal_rng(n_mean, n_std);
            while(n_posterior[i] < 0){
                n_posterior[i] = normal_rng(n_mean, n_std);
            }
            a_posterior[i] = normal_rng(a_mean, a_std);
            while(a_posterior[i] < 0){
                a_posterior[i] = normal_rng(a_mean, a_std);
            }

            b_param_posterior[i] = sersic_b_parameter(n_posterior[i]);
            pre_term_posterior[i] = graham_preterm(g_posterior[i], a_posterior[i], n_posterior[i], b_param_posterior[i], r_b_posterior[i], Re_posterior[i]);
        }

        // sample posterior
        for(i in 1:N_tot){
            log10_surf_rho_posterior[i] = normal_rng(graham_surf_density(R[i], pre_term_posterior[group_id[i]], g_posterior[group_id[i]], a_posterior[group_id[i]], r_b_posterior[group_id[i]], n_posterior[group_id[i]], b_param_posterior[group_id[i]], Re_posterior[group_id[i]], log10_I_b_posterior[group_id[i]]), err);
        }
    }


    // determine log likelihood function
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_calc[i], err);
    }

}