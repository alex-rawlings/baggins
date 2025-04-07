functions {
    real sersic_profile(real x, real y, real x0, real y0, real log_Re, real log_n, real q, real theta, real log_Ie) {
        real Re = exp(log_Re);
        real n = exp(log_n);
        real Ie = exp(log_Ie);
        
        real bn = 1.9992 * n - 0.3271;  // Approximation of bn for Sersic index

        // Coordinate transformation
        real x_prime = (x - x0) * cos(theta) + (y - y0) * sin(theta);
        real y_prime = -(x - x0) * sin(theta) + (y - y0) * cos(theta);

        real r = sqrt(x_prime^2 + (y_prime^2 / (q^2)));  // Elliptical radius
        return log(Ie) - bn * (pow(r / Re, 1 / n) - 1);
    }
}

data {
    int<lower=1> N;  // Image height
    int<lower=1> M;  // Image width
    matrix[N, M] log_image;  // Logarithm of the image data
}

parameters {
    real log_Ie;  // Log effective intensity
    real<lower=1, upper=5> log_Re;  // Log effective radius
    real<lower=0, upper=3> log_n;   // Log Sersic index
    real<lower=0, upper=1> q;  // Axis ratio
    real<lower=0, upper=pi()> theta;  // Position angle in radians
    real<lower=1, upper=M> x0;  // X center
    real<lower=1, upper=N> y0;  // Y center
    real<lower=0> sigma;  // Noise standard deviation
}

transformed parameters {
    matrix[N, M] log_model_image;
    
    // Vectorized computation of the model
    for (i in 1:N) {
        for (j in 1:M) {
            log_model_image[i, j] = sersic_profile(j, i, x0, y0, log_Re, log_n, q, theta, log_Ie);
        }
    }
}

model {
    // Priors (weakly informative)
    log_Ie ~ normal(0, 1);
    log_Re ~ normal(log(10), 1);
    log_n ~ normal(log(2), 0.5);
    q ~ beta(2, 2);
    theta ~ uniform(0, pi());
    x0 ~ uniform(1, M);
    y0 ~ uniform(1, N);
    sigma ~ normal(0, 0.5);
    
    // Likelihood (in log space)
    for (i in 1:N) {
        for (j in 1:M) {
            log_image[i, j] ~ normal(log_model_image[i, j], sigma);
        }
    }
}
