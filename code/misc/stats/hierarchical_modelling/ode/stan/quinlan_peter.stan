functions {
    #include funcs.stan
}


data {
    // number of simulation children
    int<lower=1> N_child;
    // total number of points: N_child x points/child
    int<lower=1> N_tot;
    // id of the child simulation that each point belongs to
    array[N_tot] int<lower=1, upper=N_child> child_id;
    // number of pointer per child
    array[N_child] int<lower=1> N_per_child;
    // vector of time values
    array[N_tot] real t;
    // vector of orbital energy values
    vector[N_tot] E;
    // vector of semimajor axis values
    //vector[N_tot] a;
    // vector of eccentricity
    vector[N_tot] ecc;
    // vector of bh1 mass
    vector[N_tot] mass1;
    // vector of bh2 mass
    vector[N_tot] mass2;
}


parameters {
    /**** parameters of the parent distributions ****/
    // parameters of dE/dt relation
    real Hp_s_mu;
    real<lower=0> Hp_s_tau;
    real<lower=0> sigma_E;

    // parameters for each child
    vector[N_child] Hp_s;

    // parameters for de/dt relation
    real K_mu;
    real<lower=0> K_tau;
    real<lower=0> sigma_e;

    // parameters for each child
    vector[N_child] K;
}



model {

    /**** sample prior distributions ****/
    // dE/dt relation
    Hp_s_mu ~ normal(0, 1);
    Hp_s_tau ~ cauchy(0, 1);
    sigma_E ~ cauchy(0, 10);

    // de/dt relation
    K_mu ~ normal(0.01, 0.1);
    K_tau ~ cauchy(0, 1);
    sigma_e ~ cauchy(0, 1);

    // connection to latent parameters
    Hp_s ~ normal(Hp_s_mu, Hp_s_tau);
    K ~ normal(K_mu, K_tau);

    //array[2] vector[N_tot] energy_ecc;
    //vector[2] energy_ecc[N_tot];
    array[N_tot] vector[2] energy_ecc;
    vector[2] Ee0;
    int start_idx;
    int end_idx;
    start_idx = 1;
    for(i in 1:N_child){
        end_idx = start_idx + N_per_child[i]-1;    // end index for this child
        array[N_per_child[i]-1] vector[2] temp;
        Ee0[1] = E[start_idx];
        Ee0[2] = ecc[start_idx];
        energy_ecc[start_idx,1] = Ee0[1];
        energy_ecc[start_idx,2] = Ee0[2];
        // TODO get indexing correct here!!!
        //energy_ecc[:, (start_idx+1):end_idx] = ode_rk45(peter_quinlan, Ee0, t[start_idx], t[(start_idx+1):end_idx], mass1[1], mass2[1], Hp_s[i], K[i]);
        temp = ode_rk45(peter_quinlan, Ee0, t[start_idx], t[start_idx+1:end_idx], mass1[1], mass2[1], Hp_s[i], K[i]);
        for(j in 1:N_per_child[i]-1){
            energy_ecc[start_idx+j, 1] = temp[j,1];
            energy_ecc[start_idx+j, 2] = temp[j,2];
        }
        start_idx = end_idx+1;    // index of the end of this child + 1 for next
    }

    // likelihood
    E ~ normal(energy_ecc[:,1], sigma_E);
    ecc ~ normal(energy_ecc[:,2], sigma_e);

}