functions{
    #include funcs_graham.stan
}


data{

    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // vector of radial values
    array[N_tot] real<lower=0.0> R;
    // transition index
    real<lower=0.0> a;
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
    real<lower=0.0> r_b_a;
    real<lower=0.0> r_b_b;

    // effective radius
    real<lower=0.0> Re_a;
    real<lower=0.0> Re_b;

    // break radius density
    real<lower=0.0> I_b_a;
    real<lower=0.0> I_b_b;

    // inner slope, gamma in paper
    real<lower=0.0> g_a;
    real<lower=0.0> g_b;

    // sersic parameter n
    real<lower=0.0> n_a;
    real<lower=0.0> n_b;


    /****** latent parameters for each child ******/
    array[N_child] real<lower=0.0> r_b;
    array[N_child] real<lower=0.0> Re;
    array[N_child] real<lower=0.0> I_b;
    array[N_child] real<lower=0.0> g;
    array[N_child] real<lower=0.0> n;

}


transformed parameters {
    
    array[N_tot] real log10_surf_rho_true;

    // connection between radius and log10 of surface density
    for(i in 1:N_child){
        log10_surf_rho_true[(N_per_child*(i-1)+1):(N_per_child*i)] = log10_I(N_per_child, R[(N_per_child*(i-1)+1):(N_per_child*i)], I_b[i], g[i], a, r_b[i], Re[i], n[i]);
    }
    
}


model{

    /****** sample prior distributions ******/
    r_b_a ~ gamma(3.0, 2.0);
    r_b_b ~ gamma(10.0, 8.0);

    Re_a ~ gamma(30.0, 2.0);
    Re_b ~ gamma(40.0, 2.0);

    I_b_a ~ gamma(20.0, 4.0);
    I_b_b ~ gamma(20.0, 2.0);

    g_a ~ gamma(1.0, 2.0);
    g_b ~ gamma(4.0, 2.0);

    n_a ~ gamma(16.0, 2.0);
    n_b ~ gamma(4.0, 2.0);

    /****** connection to latent parameters for each child ******/
    r_b ~ gamma(r_b_a, r_b_b);
    Re ~ gamma(Re_a, Re_b);
    I_b ~ gamma(I_b_a, I_b_b);
    g ~ gamma(g_a, g_b);
    n ~ gamma(n_a, n_b);

    /****** sample likelihood ******/
    log10_surf_rho ~ normal(log10_surf_rho_true, log10_surf_rho_err);

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

    log10_surf_rho_posterior = log10_I(N_tot, R, I_b_posterior, g_posterior, a, r_b_posterior, Re_posterior, n_posterior);

    /****** determine log likelihood function ******/
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_true[i], log10_surf_rho_err[i]);
    }

}

