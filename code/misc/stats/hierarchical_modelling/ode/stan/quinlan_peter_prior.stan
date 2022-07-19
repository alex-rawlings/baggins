functions{
    #include funcs.stan
}

data{

    // number of simulation children
    int<lower=1> N_child;
    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // number of pointer per child
    array[N_child] int<lower=1> N_per_child;
    // id of the child simulation that each point belongs to
    array[N_tot] int<lower=1, upper=N_child> child_id;
    // vector of time values
    array[N_tot] real<lower=0> t;
    // vector of BH1 masses 
    vector[N_tot] mass1;
    // vector of BH2 masses
    vector[N_tot] mass2;
    // initial energy values
    vector[N_child] E0;
    // initial eccentricity values
    vector[N_child] ecc0;
}


generated quantities {

    /**** parameters of the parent distributions ****/
    // parameters of dE/dt relation
    real Hp_s_mu;
    real<lower=0> Hp_s_tau;
    real<lower=0> sigma_E;

    // latent parameters for each child
    vector[N_child] Hp_s;

    // parameters for de/dt relation
    real K_mu;
    real<lower=0> K_tau;
    real<lower=0> sigma_e;

    // latent parameters for each child
    vector[N_child] K;

    // forward folded values
    array[N_tot] real E;
    array[N_tot] real ecc;

    
    /**** sample variables ****/
    // dE/dt relation
    Hp_s_mu = normal_rng(0, 1);
    Hp_s_tau = cauchy_rng(0, 1);
    sigma_E = cauchy_rng(0, 1);

    // de/dt relation
    K_mu = normal_rng(0.01, 0.1);
    K_tau = cauchy_rng(0, 1);
    sigma_e = cauchy_rng(0, 1);

    // connection to latent parameters
    for(n in 1:N_child){
        Hp_s[n] = normal_rng(Hp_s_mu, Hp_s_tau);
        K[n] = normal_rng(K_mu, K_tau);
    }

    // solve ODE
    array[N_tot] vector[2] energy_ecc;
    vector[2] Ee0;
    int start_idx;
    int end_idx;
    start_idx = 1;
    for(i in 1:N_child){
        end_idx = start_idx + N_per_child[i]-1;    // end index for this child
        array[N_per_child[i]-1] vector[2] temp;
        Ee0[1] = E0[i];
        Ee0[2] = ecc0[i];
        energy_ecc[start_idx,1] = Ee0[1];
        energy_ecc[start_idx,2] = Ee0[2];
        //temp = ode_rk45_tol(peter_quinlan, Ee0, t[start_idx], t[start_idx+1:end_idx], 1e4, 1e4, 1000000, mass1[1], mass2[1], Hp_s[i], K[i]);
        temp = ode_rk45(peter_quinlan, Ee0, t[start_idx], t[start_idx+1:end_idx], mass1[1], mass2[1], Hp_s[i], K[i]);
        // TODO print values from ode, make sure not returning nan
        //print(peter_quinlan())
        for(k in 1:(N_per_child[i]-1)){
            print(temp[k,1], temp[k,2]);
        }
        for(j in 1:N_per_child[i]-1){
            /*if(is_nan(temp[j,1]) || is_nan(temp[j, 2])){
                reject("Nan detected, i=",i," j=",j);
                print(temp);
            }*/
            energy_ecc[start_idx+j, 1] = temp[j,1];
            energy_ecc[start_idx+j, 2] = 1.0; //temp[j,2];
        }
        start_idx = end_idx+1;    // index of the end of this child + 1 for next
    }
    E = normal_rng(energy_ecc[:,1], sigma_E);
    ecc = normal_rng(energy_ecc[:,2], sigma_e);


}