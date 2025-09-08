vector abg_density_vec(
    vector r,
    real log10rhoS,
    real rS,
    real a,
    real b,
    real g
){
    return log10rhoS - g * log10(r ./ rS) + (g - b) / a * log10(1 + pow(r ./ rS, a));
}