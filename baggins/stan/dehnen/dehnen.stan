functions {
    vector log10_dehnen(vector r, real a, real g, real M){
        return log10((3. - g) .* M ./ (4. * pi()) .* a ./ (pow(r, g) .* pow((r + a), (4. - g))));
    }
}

data {
    int<lower=1> N;
    vector<lower=0>[N] r;
    vector<lower=0>[N] density;
    real<lower=0> mass;

    // OOS points
    int<lower=1> N_OOS;
    vector<lower=0, upper=max(r)>[N_OOS] r_OOS;
}

transformed data {
    vector[N] log10_density = log10(density);
    int N_GQ = N + N_OOS;
    vector<lower=0, upper=max(r)>[N_GQ] r_GQ = append_row(r, r_OOS);
}

parameters {
    real<lower=0> a;
    real<lower=0, upper=3> g;
    real<lower=0> err;
}

transformed parameters {
    array[3] real lprior;
    lprior[1] = normal_lpdf(a | 0, 1);
    lprior[2] = normal_lpdf(g | 1, 1);
    lprior[3] = normal_lpdf(err | 0, 2);
}

model{
    target += sum(lprior);
    target += normal_lpdf(log10_density | log10_dehnen(r, a, g, mass), err);
}

generated quantities {
    vector[N_GQ] density_posterior;
    density_posterior = to_vector(normal_rng(pow(10., to_array_1d(log10_dehnen(r_GQ, a, g, mass))), err));
}