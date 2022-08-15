functions{
    #include funcs_graham
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
    array[N_tot] real surf_rho;
    // array of log surface density value errors
    array[N_tot] real<lower=0.0> surf_rho_err;
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