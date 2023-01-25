functions{
    #include funcs_graham.stan
    #include funcs_general.stan

    real partial_sum(array[] real y_slice, 
                     int start, int end,
                     array[] real y_true,
                     array[] real y_err){
                        return normal_lpdf(y_slice | y_true[start:end], y_err[start:end]);
                     }
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
    //vector[N_tot] log10_surf_rho;
    array[N_tot] real log10_surf_rho;
    // array of log surface density value errors
    // vector<lower=0.0>[N_tot] log10_surf_rho_err;

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

    /****** radially dependent variance for each child ******/
    array[N_tot] real<lower=0.0> log10_surf_rho_err;

}


transformed parameters {
    
    array[N_tot] real log10_surf_rho_true;

    // connection between radius and log10 of surface density
    for(i in 1:N_child){
        log10_surf_rho_true[(N_per_child*(i-1)+1):(N_per_child*i)] = log10_I(N_per_child, R[(N_per_child*(i-1)+1):(N_per_child*i)], I_b[i], g[i], a[i], r_b[i], Re[i], n[i]);
    }
    
    // conversion from variance to std dev
    real<lower=0.0> r_b_std = sqrt(r_b_var);
    real<lower=0.0> Re_std = sqrt(Re_var);
    real<lower=0.0> I_b_std = sqrt(I_b_var);
    real<lower=0.0> g_std = sqrt(g_var);
    real<lower=0.0> n_std = sqrt(n_var);
    real<lower=0.0> a_std = sqrt(a_var);

}


model{

    /****** sample prior distributions ******/
    // y ~ distribution_name(a, b) is same as
    // target += distribution_name_lupdf(y | a, b);
    // second method allows for better control, first method allows for easier 
    // use of truncated distributions
    
    target += normal_lpdf(r_b_mean | 0.1, 5.0);
    if (r_b_mean < 0.0){
        target += negative_infinity();
    }else{
        target += -normal_lccdf(0.0 | 0.1, 5.0);
    }

    target += cauchy_lpdf(r_b_var | 0.0, 100.0);
    if (r_b_var < 0.0){
        target += negative_infinity();
    }else{
        target += -cauchy_lccdf(0.0 | 0.0, 100.0);
    }

    target += normal_lpdf(Re_mean | 7.0, 5.0);
    if (Re_mean < 0.0){
        target += negative_infinity();
    }else{
        target += -normal_lccdf(0.0 | 7.0, 5.0);
    }

    target += cauchy_lpdf(Re_var | 0.0, 100.0);
    if (Re_var < 0.0){
        target += negative_infinity();
    }else{
        target += -cauchy_lccdf(0.0 | 0.0, 100.0);
    }

    target += normal_lpdf(I_b_mean | 1.0, 10.0);
    if (I_b_mean < 0.0){
        target += negative_infinity();
    }else{
        target += -normal_lccdf(0.0 | 1.0, 10.0);
    }

    target += cauchy_lpdf(I_b_var | 0.0, 100.0);
    if (I_b_var < 0.0){
        target += negative_infinity();
    }else{
        target += -cauchy_lccdf(0.0 | 0.0, 100.0);
    }

    target += normal_lpdf(g_mean | 0.1, 1.0);
    if (g_mean < 0.0){
        target += negative_infinity();
    }else{
        target += -normal_lccdf(0.0 | 0.1, 1.0);
    }

    target += cauchy_lpdf(g_var | 0.0, 100.0);
    if (g_var < 0.0){
        target += negative_infinity();
    }else{
        target += -cauchy_lccdf(0.0 | 0.0, 100.0);
    }

    target += normal_lpdf(n_mean | 4.0, 5.0);
    if (n_mean < 0.0){
        target += negative_infinity();
    }else{
        target += -normal_lccdf(0.0 | 4.0, 5.0);
    }

    target += cauchy_lpdf(n_var | 0.0, 100.0);
    if (n_var < 0.0){
        target += negative_infinity();
    }else{
        target += -cauchy_lccdf(0.0 | 0.0, 100.0);
    }

    target += normal_lpdf(a_mean | 10.0, 10.0);
    if (a_mean < 0.0){
        target += negative_infinity();
    }else{
        target += -normal_lccdf(0.0 | 10.0, 10.0);
    }

    target += cauchy_lpdf(a_var | 0.0, 100.0);
    if (a_var < 0.0){
        target += negative_infinity();
    }else{
        target += -cauchy_lccdf(0.0 | 0.0, 100.0);
    }

    // variance (note this is an array)
    target += cauchy_lpdf(log10_surf_rho_err | 0.0, 100.0);
    for(i in 1:N_tot){
        if (log10_surf_rho_err[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -cauchy_lccdf(0.0 | 0.0, 100.0);
        }
    }

    /****** connection to latent parameters for each child ******/
    target += normal_lpdf(r_b | r_b_mean, r_b_std);

    target += normal_lpdf(Re | Re_mean, Re_std);

    target += normal_lpdf(I_b | I_b_mean, I_b_std);

    target += normal_lpdf(g | g_mean, g_std);

    target += normal_lpdf(n | n_mean, n_std);

    target += normal_lpdf(a | a_mean, a_std);

    for(i in 1:N_child){
        if(r_b[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -normal_lccdf(0.0 | r_b_mean, r_b_std);
        }
        if(Re[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -normal_lccdf(0.0 | Re_mean, Re_std);
        }
        if(I_b[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -normal_lccdf(0.0 | I_b_mean, I_b_std);
        }
        if(g[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -normal_lccdf(0.0 | g_mean, g_std);
        }
        if(n[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -normal_lccdf(0.0 | n_mean, n_std);
        }
        if(a[i] < 0.0){
            target += negative_infinity();
        }else{
            target += -normal_lccdf(0.0 | a_mean, a_std);
        }
    }

    /****** sample likelihood ******/
    //target += normal_lupdf(log10_surf_rho | log10_surf_rho_true, log10_surf_rho_err);
    target += reduce_sum(partial_sum, log10_surf_rho, 1, log10_surf_rho_true, sqrt(log10_surf_rho_err));

}


generated quantities {

    /****** posterior parameters ******/
    real r_b_posterior = LB_normal_rng(r_b_mean, r_b_var, 0.0);
    real Re_posterior = LB_normal_rng(Re_mean, Re_var, 0.0);
    real I_b_posterior = LB_normal_rng(I_b_mean, I_b_var, 0.0);
    real g_posterior = LB_normal_rng(g_mean, g_var, 0.0);
    real n_posterior = LB_normal_rng(n_mean, n_var, 0.0);
    real a_posterior = LB_normal_rng(a_mean, a_var, 0.0);
    
    /****** posterior predictive values ******/
    array[N_tot] real log10_surf_rho_posterior;
    array[N_tot] real surf_rho_posterior;

    log10_surf_rho_posterior = log10_I(N_tot, R, I_b_posterior, g_posterior, a_posterior, r_b_posterior, Re_posterior, n_posterior);
    surf_rho_posterior = pow(10, log10_surf_rho_posterior);

    /****** determine log likelihood function ******/
    vector[N_tot] log_lik;
    for(i in 1:N_tot){
        log_lik[i] = normal_lpdf(log10_surf_rho[i] | log10_surf_rho_true[i], log10_surf_rho_err[i]);
    }

    /****** determine log prior sensitivity ******/
    vector[12] lprior;

    lprior[1] = normal_lpdf(r_b_mean | 0.1, 0.2);
    if(lprior[1] < 0.0){
        lprior[1] = negative_infinity();
    }else{
        lprior[1] += -normal_lccdf(0.0 | 0.1, 0.2);
    }

    lprior[2] = cauchy_lpdf(r_b_var | 0, 100);
    if(lprior[2] < 0.0){
        lprior[2] = negative_infinity();
    }else{
        lprior[2] += -cauchy_lccdf(0.0 | 0, 100);
    }

    lprior[3] = normal_lpdf(Re_mean | 7.0, 2.0);
    if(lprior[3] < 0.0){
        lprior[3] = negative_infinity();
    }else{
        lprior[3] += -normal_lccdf(0.0 | 7.0, 2.0);
    }

    lprior[4] = cauchy_lpdf(Re_var | 0, 100);
    if(lprior[4] < 0.0){
        lprior[4] = negative_infinity();
    }else{
        lprior[4] += -cauchy_lccdf(0.0 | 0, 100);
    }

    lprior[5] = normal_lpdf(I_b_mean | 1.0, 3.0);
    if(lprior[5] < 0.0){
        lprior[5] = negative_infinity();
    }else{
        lprior[5] += -normal_lccdf(0.0 | 1.0, 3.0);
    }

    lprior[6] = cauchy_lpdf(I_b_var | 0, 100);
    if(lprior[6] < 0.0){
        lprior[6] = negative_infinity();
    }else{
        lprior[6] += -cauchy_lccdf(0.0 | 0, 100);
    }

    lprior[7] = normal_lpdf(g_mean | 0.1, 1.0);
    if(lprior[7] < 0.0){
        lprior[7] = negative_infinity();
    }else{
        lprior[7] += -normal_lccdf(0.0 | 0.1, 1.0);
    }

    lprior[8] = cauchy_lpdf(g_var | 0, 100);
    if(lprior[8] < 0.0){
        lprior[8] = negative_infinity();
    }else{
        lprior[8] += -cauchy_lccdf(0.0 | 0, 100);
    }

    lprior[9] = normal_lpdf(n_mean | 4.0, 2.0);
    if(lprior[9] < 0.0){
        lprior[9] = negative_infinity();
    }else{
        lprior[9] += -normal_lccdf(0.0 | 4.0, 2.0);
    }

    lprior[10] = cauchy_lpdf(n_var | 0, 100);
    if(lprior[10] < 0.0){
        lprior[10] = negative_infinity();
    }else{
        lprior[10] += -cauchy_lccdf(0.0 | 0, 100);
    }

    lprior[11] = normal_lpdf(a_mean | 10.0, 3.0);
    if(lprior[11] < 0.0){
        lprior[11] = negative_infinity();
    }else{
        lprior[11] += -normal_lccdf(0.0 | 10.0, 3.0);
    }

    lprior[12] = cauchy_lpdf(a_var | 0, 100);
    if(lprior[12] < 0.0){
        lprior[12] = negative_infinity();
    }else{
        lprior[12] += -cauchy_lccdf(0.0 | 0, 100);
    }

}

