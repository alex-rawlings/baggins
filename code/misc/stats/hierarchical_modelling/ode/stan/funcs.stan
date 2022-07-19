vector peter_quinlan(
                    real t,
                    vector y,
                    real m1,
                    real m2,
                    real Hp_s,
                    real K
){
    vector[2] dydt;
    real M = m1 + m2;
    real m1m2 = m1 * m2;
    //real mu = m1 * m2 / M;
    // convert to semimajor axis, prevent zero division
    real a = -0.5 * M / (y[1]+1e-15);
    real e2 = square(y[2]);
    
    // energy
    // original formulation, using a
    //dydt[1] = -0.5 * M * mu * Hp_s - 32.0/5.0 * square(m1*m2) * M / (pow(a,5) * pow((1.0 - e2),3.5)) * (1.0 + 73.0/24.0 * e2 + 37.0/96.0 * square(e2));
    dydt[1] = -0.5 * m1m2 * Hp_s + 1024.0/5.0 * M / (pow((m1m2), 3.0)) * pow(y[1], 5.0) * (1.0 + 73.0/24.0*e2 + 37.0/96.0*square(e2)) / (pow((1.0-e2), 3.5));

    // eccentricity
    // original formulation, using a
    //dydt[2] = K * Hp_s * a - 304.0/15.0 * m1*m2*M / (pow(a,4) * pow((1-e2), 2.5)) * y[2] * (1.0 + 121.0/304.0 * e2);
    dydt[2] = K * Hp_s * a - 4864.0/15.0 * M / (pow(m1m2, 3.0) * pow((1.0-e2), 2.5)) * pow(y[1], 4.0) * y[2] * (1.0 + 121.0/304.0 * e2);

    // prevent nan values
    /*for(i in 1:2){
        if(is_nan(dydt[i])){
            dydt[i] = 10.0;
        }
    }*/
    return dydt;
}