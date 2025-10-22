vector abg_density_vec(
    vector r,
    real log10rhoS,
    real log10rS,
    real log10a,
    real b,
    real g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    real a = pow(10, log10a);
    return log10rhoS - g * log10(x) + (g - b) / a * log10(1 + pow(x, a));
}


vector abg_density_vec(
    vector r,
    vector log10rhoS,
    vector log10rS,
    vector log10a,
    vector b,
    vector g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    vector[size(log10a)] a = pow(10, log10a);
    return log10rhoS - g .* log10(x) + (g - b) ./ a .* log10(1 + pow(x, a));
}


vector radially_vary_err(vector r, real e0, real ek, real rp){
    // radially vary error
    // e0: error at pivot radius
    // ek: error gradient, ek>0 -> grows with r, ek<0 -> shrinks with r
    // rp: pivot radius
    return e0 * pow(r / rp, ek);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10a, vector b, vector g, vector s, array[] int cidx){
    return normal_lpdf(y_slice | abg_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10a[cidx[start:end]],
                    b[cidx[start:end]],
                    g[cidx[start:end]]),
                    s[start:end]);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10a, vector b, vector g, real s, array[] int cidx){
    return normal_lpdf(y_slice | abg_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10a[cidx[start:end]],
                    b[cidx[start:end]],
                    g[cidx[start:end]]),
                    s);
}