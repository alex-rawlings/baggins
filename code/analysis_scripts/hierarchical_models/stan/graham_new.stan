functions{
    #include funcs_graham.stan
}


data{

    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // vector of radial values
    array[N_tot] real<lower=0.0> R;
    // number of samples from the population
    int<lower=1> N_child;
    // id of the sample that each point belongs to
    array[N_tot] int<lower=1, upper=N_child> child_id;
    // array of log surface density values
    vector[N_tot] log10_surf_rho;
    // array of log surface density value errors
    vector<lower=0.0>[N_tot] log10_surf_rho_err;

}


transformed data {

    // number of observations per child, assuming the same number of 
    // observations per child
    int<lower=1> N_per_child = N_tot %/% N_child; 

}


parameters{

    /****** parameters of the parent distributions ******/
    // break radius
    real<lower=0.0> r_b_mean;
    real<lower=0.0> r_b_var;

    // effective radius
    real<lower=0.0> Re_mean;
    real<lower=0.0> Re_var;

    // break radius density
    real<lower=0.0> I_b_mean;
    real<lower=0.0> I_b_var;

    // inner slope, gamma in paper
    real<lower=0.0> g_mean;
    real<lower=0.0> g_var;

    // sersic parameter n
    real<lower=0.0> n_mean;
    real<lower=0.0> n_var;

    // transition index
    real<lower=0.0> a_mean;
    real<lower=0.0> a_var;


    /****** latent parameters for each child ******/
    array[N_child] real<lower=0.0> r_b;
    array[N_child] real<lower=0.0> Re;
    array[N_child] real<lower=0.0> I_b;
    array[N_child] real<lower=0.0> g;
    array[N_child] real<lower=0.0> n;
    array[N_child] real<lower=0.0> a;

}


transformed parameters {
    
    array[N_tot] real log10_surf_rho_true;

    // connection between radius and log10 of surface density
    for(i in 1:N_child){
        log10_surf_rho_true[(N_per_child*(i-1)+1):(N_per_child*i)] = log10_I(N_per_child, R[(N_per_child*(i-1)+1):(N_per_child*i)], I_b[i], g[i], a[i], r_b[i], Re[i], n[i]);
    }
    
}


model{

    /****** sample prior distributions ******/
    target += normal_lpdf(r_b_mean | 0.1, 0.2);
    target += inv_chi_square_lpdf(r_b_var | 10);

    target += normal_lpdf(Re_mean | 7.0, 2.0);
    target += inv_chi_square_lpdf(Re_var | 10);

    target += normal_lpdf(I_b_mean | 1.0, 3.0);
    target += inv_chi_square_lpdf(I_b_var | 10);

    target += normal_lpdf(g_mean | 0.1, 1.0);
    target += inv_chi_square_lpdf(g_var | 10);

    target += normal_lpdf(n_mean | 4.0, 2.0);
    target += inv_chi_square_lpdf(n_var | 10);

    target += normal_lpdf(a_mean | 10.0, 3.0);
    target += inv_chi_square_lpdf(a_var | 10);

    /****** connection to latent parameters for each child ******/
    target += normal_lpdf(r_b | r_b_mean, r_b_var);
    target += normal_lpdf(Re | Re_mean, Re_var);
    target += normal_lpdf(I_b | I_b_mean, I_b_var);
    target += normal_lpdf(g | g_mean, g_var);
    target += normal_lpdf(n | n_mean, n_var);
    target += normal_lpdf(a | a_mean, a_var);

    /****** sample likelihood ******/
    target += normal_lpdf(log10_surf_rho | log10_surf_rho_true, log10_surf_rho_err);

}


generated quantities {

    /****** posterior parameters ******/
    real r_b_posterior = gamma_rng(r_b_a, r_b_b);
    real Re_posterior = gamma_rng(Re_a, Re_b);
    real I_b_posterior = gamma_rng(I_b_a, I_b_b);
    real g_posterior = gamma_rng(g_a, g_b);
    real n_posterior = gamma_rng(n_a, n_b);
    
    /****** posterior predictive values ******/
    array[N_tot] real log10_surf_rho_posterior;
    array[N_tot] real surf_rho_posterior;

    log10_surf_rho_posterior = log10_I(N_tot, R, I_b_posterior, g_posterior, a, r_b_posterior, Re_posterior, n_posterior);
    surf_rho_posterior = pow(10, log10_surf_rho_posterior);

    /****** determine log likelihood function ******/
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_true[i], log10_surf_rho_err[i]);
    }

    /****** determine log prior sensitivity ******/
    vector[12] lprior;
    lprior[1] = normal_lpdf(r_b_mean | 0.1, 0.2);
    lprior[2] = inv_chi_square_lpdf(r_b_var | 10);

    lprior[3] = normal_lpdf(Re_mean | 7.0, 2.0);
    lprior[4] = inv_chi_square_lpdf(Re_var | 10);

    lprior[5] = normal_lpdf(I_b_mean | 1.0, 3.0);
    lprior[6] = inv_chi_square_lpdf(I_b_var | 10);

    lprior[7] = normal_lpdf(g_mean | 0.1, 1.0);
    lprior[8] = inv_chi_square_lpdf(g_var | 10);

    lprior[9] = normal_lpdf(n_mean | 4.0, 2.0);
    lprior[10] = inv_chi_square_lpdf(n_var | 10);

    lprior[11] = normal_lpdf(a_mean | 10.0, 3.0);
    lprior[12] = inv_chi_square_lpdf(a_var | 10);

}

