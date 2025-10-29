// dynamics.cpp
#include <vector>
#include <cmath>
#include <numeric>  // for std::inner_product
#include <utility>  // for std::pair

#define G 44900

// ellipsoid acceleration
std::vector<double> ellipsoid_acceleration(
    const std::vector<double>& x,
    const double rho,
    const double ellipticity,
    const double m,
    const double mb
){
    std::vector<double> a(2);
    double e2s = ellipticity * ellipticity;
    double A1 = (1. - e2s) / e2s * ( 1. / (1. - e2s) - 1. / (2. * ellipticity) * log((1. + ellipticity) / (1. - ellipticity)));
    double A3 = 2. * (1. - e2s) / e2s * (1. / (2. * ellipticity) * log((1. + ellipticity) / (1. - ellipticity)) - 1.);
    a[0] = A3 * x[0];
    a[1] = A1 * x[1];
    for (size_t i = 0; i < a.size(); ++i){
        a[i] *= -2 * M_PI * G * rho * m / mb;
    }
    return a;
}

std::vector<double> dynamical_friction(
    const std::vector<double>& v,
    const double m1,
    const double m2,
    const double mbin,
    const double rho,
    const double logL,
    const double stellar_sigma
)
{
    auto set_vel = [&](double m, const std::vector<double>& v) {
        std::vector<double> v_single(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            v_single[i] = v[i] * m / mbin;

        double v_single_norm = std::sqrt(std::inner_product(
            v_single.begin(), v_single.end(), v_single.begin(), 0.0));

        return std::make_pair(v_single, v_single_norm);
    };

    auto df_taylor = [&](double m) {
        auto [v_single, v_single_norm] = set_vel(m, v);
        std::vector<double> result(v.size());
        double coeff = -8.0 * std::sqrt(M_PI) * m * rho * logL /
                       (3.0 * std::sqrt(2.0) * std::pow(stellar_sigma, 3));
        for (size_t i = 0; i < v.size(); ++i)
            result[i] = v_single[i] * coeff;
        return result;
    };

    auto df_general = [&](double m) {
        auto [v_single, v_single_norm] = set_vel(m, v);
        double X = v_single_norm / (std::sqrt(2.0) * stellar_sigma);
        double erf_term = std::erf(X);
        double exp_term = std::exp(-X * X);
        double coeff = -4.0 * M_PI * std::pow(G, 2) * m * rho * logL *
                       (erf_term - 2.0 * X / std::sqrt(M_PI) * exp_term) /
                       std::pow(v_single_norm, 3);

        std::vector<double> result(v.size());
        for (size_t i = 0; i < v.size(); ++i)
            result[i] = v_single[i] * coeff;
        return result;
    };

    // Compute norms for both masses
    auto [_, v1_norm] = set_vel(m1, v);
    auto [__, v2_norm] = set_vel(m2, v);

    // Choose approximation or full expression
    std::vector<double> df1 = (v1_norm / stellar_sigma < 1e-3)
                                  ? df_taylor(m1)
                                  : df_general(m1);
    std::vector<double> df2 = (v2_norm / stellar_sigma < 1e-3)
                                  ? df_taylor(m2)
                                  : df_general(m2);

    // Add df1 and df2
    std::vector<double> total(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        total[i] = df1[i] + df2[i];

    return total;
}

double orbital_energy(
    const std::vector<double>& x,
    const std::vector<double>& v,
    const double mbin
)
{
    // Compute r = ||x||
    double r = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));

    // Compute v^2 = v ⋅ v
    double v2 = std::inner_product(v.begin(), v.end(), v.begin(), 0.0);

    // Orbital energy
    return 0.5 * v2 - G * mbin / r;
}

double semimajor_axis(
    const std::vector<double>& x,
    const std::vector<double>& v,
    const double mbin
){
    return G * mbin / (2. * orbital_energy(x, v, mbin));
}

double df_decoupling_factor(
    const std::vector<double>& x,
    const std::vector<double>& v,
    double mbin,
    double a_hard
){
    double a = semimajor_axis(x, v, mbin);
    double cutoff_point = 2 * a_hard;
    double cutoff_scale = 0.5 * a_hard;
    return 1. / (1. + std::exp(-(a - cutoff_point) / cutoff_scale));
}


// acceleration for the system
std::vector<double> acceleration(double t,
                                 const std::vector<double>& y,
                                 const std::vector<double>& params)
{
    // unpack parameters
    size_t n = y.size() / 2;
    std::vector<double> x(y.begin(), y.begin() + n);
    std::vector<double> v(y.begin() + n, y.end());
    double m1 = params[0];
    double m2 = params[1];
    double rho = params[2];
    double ellipticity = params[3];
    double logL = params[4];
    double stellar_sigma = params[5];
    double a_hard = params[6];

    double mbin = m1 + m2;
    // Compute r = ||x||
    double r = std::sqrt(std::inner_product(x.begin(), x.end(), x.begin(), 0.0));

    std::vector<double> a(x.size());
    std::vector<double> ae1(x.size());
    ae1 = ellipsoid_acceleration(x, rho, ellipticity, m1, mbin);
    std::vector<double> ae2(x.size());
    ae2 = ellipsoid_acceleration(x, rho, ellipticity, m2, mbin);
    std::vector<double> adf(v.size());
    adf = dynamical_friction(v, m1, m2, mbin, rho, logL, stellar_sigma);
    double df_fac = df_decoupling_factor(x, v, mbin, a_hard);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = -G * mbin / pow(r, 3.) * x[i] + ae1[i] + ae2[i] + adf[i] * df_fac;

    return a;
}

// dydt function for solve_ivp (position + velocity system)
std::vector<double> dydt(double t,
                         const std::vector<double>& y,
                         const std::vector<double>& params)
{
    size_t n = y.size() / 2;
    std::vector<double> dy(y.size());

    // Split y into position and velocity
    for (size_t i = 0; i < n; ++i) {
        dy[i] = y[n + i]; // dx/dt = velocity
    }

    // Acceleration
    std::vector<double> a = acceleration(t, y, params);

    for (size_t i = 0; i < n; ++i) {
        dy[n + i] = a[i]; // dv/dt = acceleration
    }

    return dy;
}
