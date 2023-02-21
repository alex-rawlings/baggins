real binary_log10_angmom(real a, real ecc){
    return 0.5 * log10(a * (1.0 - pow(ecc, 2.0)));
}