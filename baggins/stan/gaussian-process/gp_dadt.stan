// icm_gp_peters_with_deriv_fixed.stan
// ICM multi-output GP for g_a = log a and g_e = logit e
// Predicts function and derivative (for g_a) at prediction grid tp
// Units: masses in Msun, distances in kpc, time unit is arbitrary (pass seconds_per_time_unit)

functions {
  // build 2x2 block matrix [A B; C D]
  matrix block22(matrix A, matrix B, matrix C, matrix D) {
    int nrow = rows(A) + rows(C);
    int ncol = cols(A) + cols(B);
    matrix[nrow, ncol] M;
    // top-left A
    for (i in 1:rows(A)) for (j in 1:cols(A)) M[i, j] = A[i, j];
    // top-right B
    for (i in 1:rows(B)) for (j in 1:cols(B)) M[i, cols(A) + j] = B[i, j];
    // bottom-left C
    for (i in 1:rows(C)) for (j in 1:cols(C)) M[rows(A) + i, j] = C[i, j];
    // bottom-right D
    for (i in 1:rows(D)) for (j in 1:cols(D)) M[rows(A) + i, cols(A) + j] = D[i, j];
    return M;
  }
}

data {
  // Observations for semimajor axis a(t) -- distances in kpc
  int<lower=1> Na;
  array[Na] real ta;        // times for a (in "time units")
  vector[Na] a_obs;         // observed semimajor axes (kpc, >0)

  // Observations for eccentricity e(t) (unitless, 0<e<1)
  int<lower=1> Ne;
  array[Ne] real te;        // times for e (same time units)
  vector[Ne] e_obs;         // observed eccentricities

  // Prediction times (same time units)
  int<lower=1> Np;
  array[Np] real tp;

  // Physical masses for Peters formula (in Msun)
  real<lower=0> m1_msun;
  real<lower=0> m2_msun;

  // Conversion: seconds in one time_unit (e.g., if your times are Myr, supply 1 Myr in seconds)
  real<lower=0> seconds_per_time_unit;
}

transformed data {
  // transform outputs to unconstrained spaces
  vector[Na] ga_obs; // log a
  vector[Ne] ge_obs; // logit e
  for (i in 1:Na) ga_obs[i] = log(a_obs[i]);
  for (i in 1:Ne) ge_obs[i] = logit(e_obs[i]);

  // stacked observations (order [ga; ge])
  int Nobs = Na + Ne;
  array[Nobs] real x_obs;
  vector[Nobs] y_obs;
  for (i in 1:Na) {
    x_obs[i] = ta[i];
    y_obs[i] = ga_obs[i];
  }
  for (j in 1:Ne) {
    x_obs[Na + j] = te[j];
    y_obs[Na + j] = ge_obs[j];
  }
}

parameters {
  // Shared base kernel hyperparameters (squared-exponential)
  real<lower=0> alpha;   // base GP marginal std (in transformed-output units)
  real<lower=0> rho;     // length-scale (in same time units as ta,tp)

  // Coregionalization: L_B is Cholesky factor of correlation, sB scales
  cholesky_factor_corr[2] L_B;
  vector<lower=0>[2] sB; // per-output amplitude factors (positive)

  // Observation noise in transformed spaces
  real<lower=0> sigma_a; // noise on log(a)
  real<lower=0> sigma_e; // noise on logit(e)
}

transformed parameters {
  // coregionalization matrix B = diag(sB) * (L_B * L_B') * diag(sB)
  matrix[2,2] R = L_B * L_B';
  matrix[2,2] B = diag_matrix(sB) * R * diag_matrix(sB);
}

model {
  // Priors (tune if you want)
  alpha ~ normal(0, 5);
  rho   ~ inv_gamma(3, 1);
  sB    ~ normal(0, 5);
  L_B   ~ lkj_corr_cholesky(2);
  sigma_a ~ normal(0, 1);
  sigma_e ~ normal(0, 1);

  real jitter = 1e-10;

  // Build observed-data covariance blocks using Stan's gp_exp_quad_cov
  matrix[Na, Na] Kaa = B[1,1] * gp_exp_quad_cov(ta, ta, alpha, rho);
  for (i in 1:Na) Kaa[i, i] += square(sigma_a) + jitter;

  matrix[Ne, Ne] Kee = B[2,2] * gp_exp_quad_cov(te, te, alpha, rho);
  for (i in 1:Ne) Kee[i, i] += square(sigma_e) + jitter;

  matrix[Na, Ne] Kae = B[1,2] * gp_exp_quad_cov(ta, te, alpha, rho);

  // Stack into full training covariance
  matrix[Na+Ne, Na+Ne] K = block22(Kaa, Kae, Kae', Kee);

  // zero mean GP on transformed outputs
  vector[Na+Ne] mu = rep_vector(0, Na + Ne);

  // Likelihood (analytic)
  matrix[Na+Ne, Na+Ne] L_K = cholesky_decompose(K);
  target += multi_normal_cholesky_lpdf(y_obs | mu, L_K);
}

generated quantities {
  // physical constants & conversions
  real M_sun_kg = 1.98847e30;                 // kg
  real kpc_m = 3.085677581491367e19;          // meters per kpc
  real seconds_per_tu = seconds_per_time_unit;

  //int Nobs = Na + Ne;
  real jitter = 1e-10;

  // Rebuild training covariance & cholesky (same as model)
  matrix[Na, Na] Kaa = B[1,1] * gp_exp_quad_cov(ta, ta, alpha, rho);
  for (i in 1:Na) Kaa[i, i] += square(sigma_a) + jitter;
  matrix[Ne, Ne] Kee = B[2,2] * gp_exp_quad_cov(te, te, alpha, rho);
  for (i in 1:Ne) Kee[i, i] += square(sigma_e) + jitter;
  matrix[Na, Ne] Kae = B[1,2] * gp_exp_quad_cov(ta, te, alpha, rho);
  matrix[Nobs, Nobs] K = block22(Kaa, Kae, Kae', Kee);

  matrix[Nobs, Nobs] L_K = cholesky_decompose(K);

  // K^{-1} * y via Cholesky solves
  vector[Nobs] tmp = mdivide_left_tri_low(L_K, y_obs);
  vector[Nobs] Kinv_y = mdivide_right_tri_low(tmp', L_K)';

  // -------- Precompute base kernels for predictions --------
  matrix[Na, Np] K_ta_tp = gp_exp_quad_cov(ta, tp, alpha, rho);
  matrix[Ne, Np] K_te_tp = gp_exp_quad_cov(te, tp, alpha, rho);
  matrix[Np, Np] K_tp_tp = gp_exp_quad_cov(tp, tp, alpha, rho);

  // derivative kernels (constructed from base kernel)
  matrix[Na, Np] K_ta_tp_d; // ∂/∂tp' k(ta, tp)
  matrix[Ne, Np] K_te_tp_d;
  matrix[Np, Np] K_tp_tp_d;   // ∂/∂tp' k(tp, tp')
  matrix[Np, Np] K_tp_tp_d2;  // ∂^2/∂tp ∂tp' k(tp, tp')

  for (i in 1:Na) for (j in 1:Np)
    K_ta_tp_d[i, j] = (ta[i] - tp[j]) / square(rho) * K_ta_tp[i, j];

  for (i in 1:Ne) for (j in 1:Np)
    K_te_tp_d[i, j] = (te[i] - tp[j]) / square(rho) * K_te_tp[i, j];

  for (i in 1:Np) for (j in 1:Np) {
    real d = tp[i] - tp[j];
    K_tp_tp_d[i, j] = (tp[i] - tp[j]) / square(rho) * K_tp_tp[i, j];
    K_tp_tp_d2[i, j] = (1.0 / square(rho) - square(d) / pow(rho, 4)) * K_tp_tp[i, j];
  }

  // -------- Build Kxs (Nobs x 3Np) for outputs [g_a(tp); g_e(tp); g_a'(tp)] --------
  int Npred_tot = 3 * Np;
  matrix[Nobs, 3 * Np] Kxs;
  // zero-init
  for (i in 1:Nobs) for (j in 1:3 * Np) Kxs[i, j] = 0.0;

  // Fill for ga-observed rows (1:Na)
  for (i in 1:Na) {
    for (j in 1:Np) {
      Kxs[i, j] = B[1,1] * K_ta_tp[i, j];           // cov(ga_obs, g_a(tp))
      Kxs[i, Np + j] = B[1,2] * K_ta_tp[i, j];     // cov(ga_obs, g_e(tp))
      Kxs[i, 2 * Np + j] = B[1,1] * K_ta_tp_d[i, j]; // cov(ga_obs, g_a'(tp))
    }
  }

  // Fill for ge-observed rows (Na+1 : Na+Ne)
  for (ii in 1:Ne) {
    int irow = Na + ii;
    for (j in 1:Np) {
      Kxs[irow, j] = B[2,1] * K_te_tp[ii, j];         // cov(ge_obs, g_a(tp))
      Kxs[irow, Np + j] = B[2,2] * K_te_tp[ii, j];    // cov(ge_obs, g_e(tp))
      Kxs[irow, 2 * Np + j] = B[2,1] * K_te_tp_d[ii, j]; // cov(ge_obs, g_a'(tp))
    }
  }

  // -------- Build Kss (3Np x 3Np) predictive covariance blocks --------
  matrix[3 * Np, 3 * Np] Kss;
  for (i in 1:(3 * Np)) for (j in 1:(3 * Np)) Kss[i, j] = 0.0;

  // block (gp_a, gp_a)
  for (i in 1:Np) for (j in 1:Np) Kss[i, j] = B[1,1] * K_tp_tp[i, j];
  // block (gp_a, gp_e)
  for (i in 1:Np) for (j in 1:Np) Kss[i, Np + j] = B[1,2] * K_tp_tp[i, j];
  // block (gp_a, gp_a_deriv)
  for (i in 1:Np) for (j in 1:Np) Kss[i, 2 * Np + j] = B[1,1] * K_tp_tp_d[i, j];

  // block (gp_e, gp_a)
  for (i in 1:Np) for (j in 1:Np) Kss[Np + i, j] = B[2,1] * K_tp_tp[i, j];
  // block (gp_e, gp_e)
  for (i in 1:Np) for (j in 1:Np) Kss[Np + i, Np + j] = B[2,2] * K_tp_tp[i, j];
  // block (gp_e, gp_a_deriv)
  for (i in 1:Np) for (j in 1:Np) Kss[Np + i, 2 * Np + j] = B[2,1] * K_tp_tp_d[i, j];

  // block (gp_a_deriv, gp_a)
  for (i in 1:Np) for (j in 1:Np) Kss[2 * Np + i, j] = B[1,1] * ( - K_tp_tp_d[j, i] ); // ∂/∂tp k(tp, tp') = -∂/∂tp' k(tp, tp')
  // block (gp_a_deriv, gp_e)
  for (i in 1:Np) for (j in 1:Np) Kss[2 * Np + i, Np + j] = B[2,1] * ( - K_tp_tp_d[j, i] );
  // block (gp_a_deriv, gp_a_deriv)
  for (i in 1:Np) for (j in 1:Np) Kss[2 * Np + i, 2 * Np + j] = B[1,1] * K_tp_tp_d2[i, j];

  // Symmetrize Kss to remove any tiny asymmetry
  Kss = 0.5 * (Kss + Kss');

  // add a bit of jitter
  for (i in 1:(3 * Np)) Kss[i, i] += 1e-10;

  // -------- Compute predictive mean and covariance (analytic) --------
  matrix[Nobs, 3 * Np] v = mdivide_left_tri_low(L_K, Kxs);
  matrix[Nobs, 3 * Np] Kinv_Kxs = mdivide_right_tri_low(v', L_K)';

  // predictive mean: Kxs' * K^{-1} * y_obs
  vector[3 * Np] fstar_mu;
  for (j in 1:(3 * Np)) {
    real acc = 0.0;
    for (i in 1:Nobs) acc += Kxs[i, j] * Kinv_y[i];
    fstar_mu[j] = acc;
  }

  // predictive covariance: Kss - Kxs' * K^{-1} * Kxs
  matrix[3 * Np, 3 * Np] fstar_cov = Kss;
  for (i in 1:(3 * Np)) {
    for (j in 1:(3 * Np)) {
      real acc = 0.0;
      for (k in 1:Nobs) acc += Kxs[k, i] * Kinv_Kxs[k, j];
      fstar_cov[i, j] -= acc;
    }
  }

  // Force symmetry (important) and add final jitter
  fstar_cov = 0.5 * (fstar_cov + fstar_cov');
  for (i in 1:(3 * Np)) fstar_cov[i, i] += 1e-8;

  // Draw a joint sample of predictive outputs (one sample per Stan draw)
  vector[3 * Np] fstar = multi_normal_rng(fstar_mu, fstar_cov);

  // Unpack sampled predictive vectors
  vector[Np] gp_loga;
  vector[Np] gp_logit_e;
  vector[Np] gp_loga_deriv; // derivative of log a with respect to time (in units 1/time_unit)
  for (i in 1:Np) {
    gp_loga[i] = fstar[i];
    gp_logit_e[i] = fstar[Np + i];
    gp_loga_deriv[i] = fstar[2 * Np + i];
  }

  // Transform back to physical predictions
  vector[Np] a_pred = exp(gp_loga);       // kpc
  vector[Np] e_pred = inv_logit(gp_logit_e);

  // --- Peters (GW) da/dt calculation (SI -> convert to kpc / time_unit) ---
  real M_sun = M_sun_kg;
  real G_const = 6.67430e-11;
  real c_const = 2.99792458e8;

  real m1_si = m1_msun * M_sun;
  real m2_si = m2_msun * M_sun;

  real mu_peters = (G_const * G_const * G_const * m1_si * m2_si * (m1_si + m2_si)) / pow(c_const, 5);

  vector[Np] da_dt_peters; // kpc / time_unit
  for (i in 1:Np) {
    real ai_kpc = a_pred[i];
    real ai_m = ai_kpc * kpc_m;
    real ei = e_pred[i];
    real one_me2 = 1.0 - ei * ei;
    real pref = (64.0 / 5.0) * mu_peters / ( pow(ai_m, 3) * pow(one_me2, 3.5) );
    real bracket = 1.0 + (73.0 / 24.0) * ei * ei + (37.0 / 96.0) * pow(ei, 4);
    real da_dt_SI = - pref * bracket; // m / s
    da_dt_peters[i] = da_dt_SI * seconds_per_tu / kpc_m; // kpc / time_unit
  }

  // --- GP-based total da/dt from derivative of log a: da/dt = a * d(log a)/dt ---
  vector[Np] da_dt_gp; // kpc / time_unit
  for (i in 1:Np) da_dt_gp[i] = a_pred[i] * gp_loga_deriv[i];

  // Hardening rates s = d(1/a)/dt = - (1 / a^2) * da/dt
  vector[Np] s_total;
  vector[Np] s_peters;
  vector[Np] s_other;
  for (i in 1:Np) {
    s_total[i] = - (1.0 / (a_pred[i] * a_pred[i])) * da_dt_gp[i];
    s_peters[i] = - (1.0 / (a_pred[i] * a_pred[i])) * da_dt_peters[i];
    s_other[i] = s_total[i] - s_peters[i];
  }

  // --- Numerical finite-difference derivative of input semimajor axis data (kpc / time_unit) ---
  vector[Na] da_dt_obs;
  if (Na == 1) {
    da_dt_obs[1] = 0.0;
  } else {
    da_dt_obs[1] = (a_obs[2] - a_obs[1]) / (ta[2] - ta[1]);
    for (i in 2:(Na - 1)) da_dt_obs[i] = (a_obs[i + 1] - a_obs[i - 1]) / (ta[i + 1] - ta[i - 1]);
    da_dt_obs[Na] = (a_obs[Na] - a_obs[Na - 1]) / (ta[Na] - ta[Na - 1]);
  }

  // Expose useful quantities:
  // a_pred, e_pred, da_dt_gp, da_dt_peters, s_total, s_peters, s_other, da_dt_obs
}
