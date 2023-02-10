generated quantities {
    /****** parameters of the parent distributions ******/
    // semimajor axis at time binary is hard
    real a_hard;
    real a_hard_a;
    real a_hard_b;

    // eccentricity at time binary is hard
    real e_hard;
    real e_hard_a;
    real e_hard_b;

    // general scatter
    real sigma2;

    /****** forward folded values ******/
    real log_angmom;

    /****** sample the parameters ******/
    // semimajor axis
    a_hard_a = gamma_rng(10, 10);
    a_hard_b = gamma_rng(10, 10);
    a_hard = gamma_rng(a_hard_a, a_hard_b);

    // eccentricity
    e_hard_a = gamma_rng(10, 10);
    e_hard_b = gamma_rng(10, 10);
    e_hard = beta_rng(e_hard_a, e_hard_b);

    sigma2 = gamma_rng(10.0, 10.0);

    //angmom = normal_rng(sqrt(total_mass * a_hard * (1.0 - pow(e_hard, 2.0))), sqrt(sigma2));
    log_angmom = normal_rng(0.5 * (log10(a_hard) * log10(1.0 - pow(e_hard, 2.0))), sqrt(sigma2));
}