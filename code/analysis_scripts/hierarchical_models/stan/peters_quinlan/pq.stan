functions {
    // ode system
    vector pq_ode(real t, vector y, real Hps, real K, real M1, real M2){
        real M = M1 + M2;
        vector[2] dydt;
        real e2 = pow(y[2], 2.);
        // da/dt
        dydt[1] = -Hps / (y[1] * y[1]) - 64./5. * M1 * M2 * M / (pow(y[1], 3.0) * pow(1 - e2, 3.5)) * (1 + 73./24.*e2 + 37./96.*pow(e2, 2.));
        // de/dt
        dydt[2] = -K / y[1] * dydt[1] - 304./15. * M1 * M2 * M / (pow(y[1],4.) * pow(1 - e2, 2.5)) * y[2] * (1 + 121./304. * e2);
        return dydt;
    }
}


data {
    int<lower=1> N_tot;
    real<lower=0> M1;
    real<lower=0> M2;
    array[N_tot] real<lower=0> t;
    array[N_tot] real<lower=0> a;
    array[N_tot] real<lower=0, upper=1> ecc;
}


parameters {
    real<lower=0> Hps;
    real K;
    real ah;
    real eh;
}


model {
    target += normal_lpdf(Hps | 0, 1);
    target += normal_lpdf(K | 0, 0.5);

}