functions {
    #include funcs_terzic.stan
}


data {
    
    // number of simulation children
    int<lower=1> N_child;
    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // number of points per child
    array[N_child] int<lower=1> N_per_child;
    // id of the child simulation that each point belongs to
    array[N_tot] int<lower=1, upper=N_child> child_id;
    // vector of radial values
    array[N_tot] real<lower=0> R;
    // mass to light ratio
    real<lower=0> Upsilon;

}


generated quantities {
    
    /****** parameters of the parent distributions ******/
    // break radius
    real r_b;
    real<lower=0.0> r_b_mu;
    real<lower=0.0> r_b_sigma;

    // effective radius
    real Re;
    real<lower=0.0> Re_mu;
    real<lower=0.0> Re_sigma;

    // break radius density
    real rho_b;
    real<lower=0.0> rho_b_mu;
    real<lower=0.0> rho_b_sigma;

    // inner slope, gamma in paper
    real g;
    real<lower=0.0> g_mu;
    real<lower=0.0> g_sigma;

    // sersic parameter n
    real n;
    real<lower=0.0> n_mu;
    real<lower=0.0> n_sigma;


    /****** latent parameters for each child ******/
    array[N_tot] real<lower=0.0> r_b;
    array[N_tot] real<lower=0.0> Re;
    array[N_tot] real<lower=0.0> rho_b;
    array[N_tot] real<lower=0.0> g;
    array[N_tot] real<lower=0.0> n;


    /****** forward folded values ******/
    real<lower=0.0> projected_density_sigma;
    array[N_tot] real<lower=0> projected_density;


    /****** sample the parameters ******/
    r_b_mu = normal_rng(1e-2, 1e-2);
    r_b_sigma = inv_gamma_rng(0.0, 1.0);

    Re_mu = normal_rng(7.0, 2.0);
    Re_sigma = inv_gamma_rng(0.0, 2.0);

    rho_b = normal_rng(1.0, 0.5);
    rho_b_sigma = inv_gamma_rng(0.0, 10.0);

    g_mu = normal_rng(1.5, 0.5);
    g_sigma = inv_gamma_rng(0.0, 2.0);

    n_mu = normal_rng(4.0, 0.5);
    n_sigma = inv_gamma_rng(0.0, 2.0);


    /****** connection to latent parameters ******/
    for(i in 1:N_child){
        r_b[i] = normal_rng(r_b_mu, r_b_sigma);
        Re[i] = normal_rng(Re_mu, Re_sigma);
        rho_b[i] = normal_rng(rho_b_mu, rho_b_sigma);
        g[i] = normal_rng(g_mu, g_sigma);
        n[i] = normal_rng(n_mu, n_sigma);
    }
    projected_density_sigma = inv_gamma_rng(0.0, 1.0);

    projected_density = normal_rng(I(R, g, Re, rho_b, n, Upsilon), projected_density_sigma);

}