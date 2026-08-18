vector mNFW_density_vec(
    vector r,
    real log10rhoS,
    real log10rS,
    real log10g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    real g = pow(10, log10g);
    return log10rhoS - g * log10(x) + (g - 3.) * log10(1 + x);
}


vector mNFW_density_vec(
    vector r,
    vector log10rhoS,
    vector log10rS,
    vector log10g
){
    vector[size(r)] x = r ./ pow(10, log10rS);
    vector[size(log10g)] g = pow(10, log10g);
    return log10rhoS - g .* log10(x) + (g - 3.) .* log10(1 + x);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10g, vector s, array[] int cidx){
    return normal_lpdf(y_slice | mNFW_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10g[cidx[start:end]]),
                    s[start:end]);
}


real partial_sum_hierarchy(array[] real y_slice, int start, int end, vector r, vector log10rhoS, vector log10rS, vector log10g, real s, array[] int cidx){
    return normal_lpdf(y_slice | mNFW_density_vec(
                    r[start:end],
                    log10rhoS[cidx[start:end]],
                    log10rS[cidx[start:end]],
                    log10g[cidx[start:end]]),
                    s);
}


real mNFW_density_scalar(
    real r,
    real log10rhoS,
    real log10rS,
    real log10g
){
    real x = r / pow(10, log10rS);
    real g = pow(10, log10g);
    return log10rhoS - g * log10(x) + (g - 3.) * log10(1 + x);
}


vector mNFWspike_density_vec(
    vector r,
    real log10rhoS,
    real log10rS,
    real log10g,
    real M_BH,
    real gamma_sp,
    real r_min
){
    // DM spike carved out of an mNFW halo by adiabatic (or post-merger) growth
    // of the central BH, following Alonso-Alvarez, Cline & Dewar (2024), arXiv:2401.14450.
    // The spike radius r_sp = 0.2 * r_2M uses the small-radius NFW mass expansion
    // (their Eq. 2), so it is only strictly valid for an NFW-like (g ~ 1) inner cusp.
    int N = size(r);
    vector[N] log10_rho;
    real rhoS = pow(10, log10rhoS);
    real rS = pow(10, log10rS);
    real r_sp = 0.2 * sqrt(M_BH / (pi() * rhoS * rS));
    real log10rho_sp = mNFW_density_scalar(r_sp, log10rhoS, log10rS, log10g);

    for (i in 1:N) {
        if (r[i] < r_sp) {
            log10_rho[i] = log10rho_sp + gamma_sp * log10(r_sp / fmax(r[i], r_min));
        } else {
            log10_rho[i] = mNFW_density_scalar(r[i], log10rhoS, log10rS, log10g);
        }
    }
    return log10_rho;
}