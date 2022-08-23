functions {
    #include funcs_graham.stan
}


data {
    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // vector of radial values
    array[N_tot] real<lower=0.0> R;
    // transition index
    real<lower=0.0> a;

}


generated quantities {
    
    /****** parameters of the parent distributions ******/
    // break radius
    real r_b_a;
    real r_b_b;

    // effective radius
    real Re_a;
    real Re_b;

    // break radius density
    real I_b_a;
    real I_b_b;

    // inner slope, gamma in paper
    real g_a;
    real g_b;

    // sersic parameter n
    real n_a;
    real n_b;


    /****** latent parameters ******/
    real r_b;
    real Re;
    real I_b;
    real g;
    real n;
    real projected_density_sigma;


    /****** forward folded values ******/
    array[N_tot] real projected_density;


    /****** sample the parameters ******/
    r_b_a = gamma_rng(3.0, 2.0);
    r_b_b = gamma_rng(10.0, 8.0);

    Re_a = gamma_rng(30.0, 2.0);
    Re_b = gamma_rng(40.0, 2.0);

    I_b_a = gamma_rng(20.0, 4.0);
    I_b_b = gamma_rng(20.0, 2.0);

    g_a = gamma_rng(1.0, 2.0);
    g_b = gamma_rng(4.0, 2.0);

    n_a = gamma_rng(16.0, 2.0);
    n_b = gamma_rng(4.0, 2.0);

    
    /****** connection to latent parameters ******/
    r_b = gamma_rng(r_b_a, r_b_b);
    Re = gamma_rng(Re_a, Re_b);
    I_b = gamma_rng(I_b_a, I_b_b);
    g = gamma_rng(g_a, g_b);
    n = gamma_rng(n_a, n_b);
    projected_density_sigma = gamma_rng(2,4);

    projected_density = normal_rng(log10_I(N_tot, R, I_b, g, a, r_b, Re, n), projected_density_sigma);

}

