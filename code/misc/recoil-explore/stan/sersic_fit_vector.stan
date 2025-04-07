functions {
    vector sersic_profile(int N, vector x, vector y, real x0, real y0, real log_Re, real n, real q, real theta, real log_Ie) {
        real Re = exp(log_Re);
        
        real bn = 2.0 * n - 0.33333333 + 0.009876 * inv(n);;  // Approximate bn

        // Coordinate transformation
        vector[N] x_prime = (x - x0) .* cos(theta) + (y - y0) .* sin(theta);
        vector[N] y_prime = -(x - x0) .* sin(theta) + (y - y0) .* cos(theta);

        vector[N] r = sqrt(x_prime^2 + (y_prime^2 ./ (q^2)));  // Elliptical radius
        return log_Ie - bn .* (pow(r / Re, 1 ./ n) - 1);
    }
}

data {
    int<lower=1> N_valid;  // Number of valid pixels
    vector[N_valid] x;  // X-coordinates of valid pixels
    vector[N_valid] y;  // Y-coordinates of valid pixels
    vector[N_valid] log_image;  // Log intensity values of valid pixels
}

transformed data {
    real mean_x = mean(x);
    real mean_y = mean(y);
    real upper_x = max(x);
    real upper_y = max(y);
    real mean_log_image = mean(log_image);
    real sd_log_image = sd(log_image);
}

parameters {
    real log_Ie;  // Log effective intensity
    real<lower=0, upper=2.5>log_Re;  // Log effective radius
    real<lower=0, upper=15>n;   // Sérsic index
    real<lower=0, upper=1> q;  // Axis ratio
    real<lower=0, upper=pi()> theta;  // Position angle in radians
    real<lower=1, upper=upper_x> x0;  // X center
    real<lower=1, upper=upper_y> y0;  // Y center
    real<lower=0> sigma;  // Noise standard deviation
}

transformed parameters {
    array[8] real lprior;
    lprior[1] = normal_lpdf(log_Ie | 0, 1);
    lprior[2] = rayleigh_lpdf(log_Re | 4);
    lprior[3] = normal_lpdf(n | 4, 4);
    //lprior[4] = beta_lpdf(q | 2, 2);
    lprior[4] = beta_lpdf(q | 3, 2);
    lprior[5] = uniform_lpdf(theta | 0, pi());
    lprior[6] = normal_lpdf(x0 | mean_x, 2);
    lprior[7] = normal_lpdf(y0 | mean_y, 2);
    lprior[8] = normal_lpdf(sigma | 0, 2);

    vector[N_valid] log_model_image = sersic_profile(N_valid, x, y, x0, y0, log_Re, n, q, theta, (log_Ie-mean_log_image)/sd_log_image);
}

model {
    target += sum(lprior);
    // Vectorized likelihood
    target += normal_lpdf(log_image | log_model_image*sd_log_image+mean_log_image, sigma);
}
