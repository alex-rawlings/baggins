// ode system
vector pq_ode_1(real t, vector y, real Hps, real K, real M1, real M2){
    // unit conversions to pc, Myr, Msol
    real G = 0.0045185313263043795;
    real c = 306601.39378555;
    real M = M1 + M2;
    vector[2] dydt;
    real e2 = pow(y[2], 2.);
    real G3M1M2M_per_c5 = pow(G, 3.) * M1 * M2 * M / pow(c, 5.);
    // da/dt
    // TODO * a^2, ODE solver can't handle it
    dydt[1] = -G * Hps * pow(y[1], 2.) - 64./5. * G3M1M2M_per_c5 / (pow(y[1], 3.) * pow(1 - e2, 3.5)) * (1 + 73./24.*e2 + 37./96.*pow(e2, 2.));
    // de/dt
    dydt[2] = -K / y[1] * dydt[1] - 304./15. * G3M1M2M_per_c5 / (pow(y[1],4.) * pow(1 - e2, 2.5)) * y[2] * (1 + 121./304. * e2);

    return dydt;
}


vector pq_ode(real t, vector y, real Hps, real K, real M1, real M2){
    // unit conversions to pc, Myr, Msol
    real G = 0.0045185313263043795;
    vector[1] dydt;
    // da^-1/dt
    dydt[1] = G * Hps;
    // de/dt
    dydt[2] = K * G * Hps / y[1];
    return dydt;
}