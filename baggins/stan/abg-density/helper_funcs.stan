vector abg_density_vec(
    vector r,
    real log10rhoS,
    real log10rS,
    real a,
    real b,
    real g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    return log10rhoS - g * log10(x) + (g - b) / a * log10(1 + pow(x, a));
}


vector radially_vary_err(vector r, real e0, real ek, real rp){
    // radially vary error
    // e0: error at pivot radius
    // ek: error gradient, ek>0 -> grows with r, ek<0 -> shrinks with r
    // rp: pivot radius
    return e0 * pow(r / rp, ek);
}