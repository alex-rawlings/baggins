#include <boost/math/special_functions/beta.hpp>

real sersic_b_parameter(real n){
    return 2.0 * n - 0.33333333 + 0.009876 / n;
}

real p_parameter(real n){
    return 1.0 - 0.6097/n + 0.05563/(n*n);
}

real I1(real R, real r_b, real g, real rho_b, real Upsilon){
    real term;
    real tol = 1e-15;

    if(g < tol) {
        term = sqrt(r_b * r_b - R * R);
    } else if (fabs(g - 1.0) < tol) {
        term = log((r_b + sqrt(r_b * r_b - R * R)) * 1.0 / R);
    } else if (fabs(g - 2.0) < tol) {
        term = inv(R) * asin(sqrt(1.0 - R*R / (r_b * r_b)));
    } else {
        // TODO not sure if betac() will work...
        term = 0.5 * pow(R, 1.0-g) * boost::math::betac(0.5, 0.5*(g-1.0), 1.0-(R*R)/(r_b*r_b));
    }
    return 2.0 / Upsilon * rho_b * pow(r_b, g) * term;
}

real Ie(real Re, real rho_b, real<lower=0.5, upper=10> n, real Upsilon){
    real b = sersic_b_parameter(n);
    real p = p_parameter(n);
    real rho_bar = pow((r_b / Re), p) * exp(b * pow(r_b/Re, inv(n)));
    return 2 * exp(-b) * Re * rho_b * rho_bar * tgamma(n * (3-p)) / (Upsilon * tgamma(2*n) * pow(b, n*(1-p)));
}

real I2(real R, real Re, real rho_b, real n, real Upsilon){
    return Ie(b, Re, rho_b, n, Upsilon) * exp(b) * exp(-b * pow(R/Re), inv(n));
}

real I(real R, real g, real Re, real rho_b, real n, real Upsilon){
    if(R <= r_b) {
        return I1(R, r_b, g, rho_b, Upsilon) + I2(R, Re, rho_b, n, Upsilon);
    } else {
        return I2(R, Re, rho_b, n, Upsilon);
    }
}