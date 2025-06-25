/*
Calculate the Sersic b parameter, vectorised

Parameters
----------
n: sersic index

Returns
-------
b value (approximate)
*/
vector sersic_b_parameter(vector n){
    return 2.0 * n - 0.33333333 + 0.009876 * inv(n);
}

/*
Calculated Sersic b parameter, serial
*/
real sersic_b_parameter(real n){
    return 2.0 * n - 0.33333333 + 0.009876 * inv(n);
}

real p_parameter(real n){
    return 1.0 - 0.6097 / n + 0.05563 / pow(n, 2.);
}

/*
Determine the pre-factor term of the Terzic model, serial
*/
real terzic_preterm(real g, real a, real rb, real Re, real n, real b, real p){
    return - (p-g)/a*log(2.0) + p * log(rb/Re) + b * pow((pow(2.0, inv(a))*rb/Re), inv(n));
}

/*
Calculate the Terzic profile, serial over parameters
*/
vector terzic_density_vec(
                             vector r,
                             real preterm,
                             real g,
                             real a,
                             real rb,
                             real n,
                             real b,
                             real p,
                             real Re,
                             real log10rhob){
    vector[size(r)] radial_term = (pow(r, a) + pow(rb, a)) / pow(Re, a);
    return log10rhob + (
        preterm + g/a*log(pow(r,a) + pow(rb,a)) - g*log(r) - p/a * log(radial_term) - b * pow(radial_term, n*a)
        ) / log(10.0);
}