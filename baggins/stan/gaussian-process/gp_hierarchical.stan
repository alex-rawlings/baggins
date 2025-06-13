data {
  int<lower=1> G;                  // number of groups
  array[G] int <lower=1> N;               // number of observations per group
  int<lower=1> N_total;            // total number of observations
  vector[N_total] y;              // response vector
  vector[N_total] x;              // covariate vector
  array[N_total] int <lower=1, upper=G> group_id;  // group index for each observation
}

transformed data {
  array[G+1] int pos;
  pos[1] = 1;
  for (g in 2:(G + 1)) {
    pos[g] = pos[g - 1] + N[g - 1];
  }
}

parameters {
  // Hierarchical kernel hyperparameters
  real<lower=0> mu_lengthscale;
  real<lower=0> mu_sigma_f;

  real<lower=0> sigma_lengthscale;
  real<lower=0> sigma_sigma_f;

  vector<lower=0>[G] lengthscale_raw;
  vector<lower=0>[G] sigma_f_raw;

  real<lower=0> sigma;  // observation noise

  vector[N_total] f;  // latent GP function values
}

transformed parameters {
  vector[G] lengthscale = mu_lengthscale + sigma_lengthscale * lengthscale_raw;
  vector[G] sigma_f = mu_sigma_f + sigma_sigma_f * sigma_f_raw;
}

model {
  mu_lengthscale ~ normal(1, 1);
  sigma_lengthscale ~ normal(0, 1);
  mu_sigma_f ~ normal(1, 1);
  sigma_sigma_f ~ normal(0, 1);

  lengthscale_raw ~ std_normal();
  sigma_f_raw ~ std_normal();

  sigma ~ normal(0, 1);

  int idx;
  idx = 1;

  for (g in 1:G) {
    int n = N[g];
    matrix[n, n] K;
    matrix[n, n] L_K;
    vector[n] f_sub;

    for (i in 1:n) {
      for (j in i:n) {
        real sqdist = square(x[idx + i - 1] - x[idx + j - 1]);
        K[i, j] = square(sigma_f[g]) * exp(-0.5 * sqdist / square(lengthscale[g]));
        if (i != j) {
          K[j, i] = K[i, j];
        }
      }
      K[i, i] += 1e-6;  // jitter for numerical stability
    }

    L_K = cholesky_decompose(K);
    f_sub = segment(f, idx, n);
    f_sub ~ multi_normal_cholesky(rep_vector(0, n), L_K);
    idx += n;
  }

  y ~ normal(f, sigma);
}

generated quantities {
  vector[N_total] y_rep;
  for (n in 1:N_total) {
    y_rep[n] = normal_rng(f[n], sigma);
  }
}
