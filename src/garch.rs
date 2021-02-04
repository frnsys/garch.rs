use rand::Rng;
use super::util;
use super::error::GarchError;
use rand_distr::{Normal, Distribution};
use rustimization::lbfgsb_minimizer::Lbfgsb;
use finitediff::FiniteDiff;

/// Compute $sigma_2$ over a time series of residuals
pub fn garch_recursion(omega: f64, alpha: &[f64], beta: &[f64], eps: &[f64]) -> Vec<f64> {
    let mut sigma_2 = Vec::with_capacity(eps.len());
    let m = util::mean(eps);
    let init_sigma_2 = eps.iter().fold(0., |acc, e| acc + (e-m).powf(2.))/(eps.len()-1) as f64;

    for i in 0..eps.len() {
        if i < alpha.len() || i < beta.len() {
            sigma_2.push(init_sigma_2);
        } else {
            let next = predict_next(omega, alpha, beta, &eps[..i], &sigma_2[..i]);
            sigma_2.push(next);
        }
    }
    sigma_2
}


/// Calculating $sigma^2_t$:
/// $$ \sigma^2_t = \omega + \sum^p_{i=0} \alpha_i \epsilon^2_{t-1-i} + \sum^{q}_{i=0} \beta_i \sigma^2_{t-1-i} $$
///
/// If there isn't enough history to `predict_next`,
/// then initialize $sigma^2_t$ as the sample variance.
/// This isn't the only way to initialize these values, e.g. could also initialize
/// to the unconditional variance:
/// $$ \frac{\omega}{1 - (\sum \alpha + \sum \beta)} $$
/// but that is less stable (e.g. if $\alpha$ and $\beta$ sum to 1 or greater)
fn predict_next(omega: f64, alpha: &[f64], beta: &[f64], eps: &[f64], sigma_2: &[f64]) -> f64 {
    let n_e = eps.len();
    let n_s = sigma_2.len();
    let residual_term = alpha.iter().enumerate().fold(0., |acc, (j, a)| {
        let t = n_e - (j + 1);
        acc + (a * eps[t].powf(2.))
    });
    let volatility_term = beta.iter().enumerate().fold(0., |acc, (j, b)| {
        let t = n_s - (j + 1);
        acc + (b * sigma_2[t])
    });
    omega + residual_term + volatility_term
}


/// The model is fit with MLE
/// (though we use negative log likelihood b/c we optimize by minimization)
///
/// Using the standard normal log likelihood:
/// $$ -\sum^T_t \frac{1}{2} (-\log 2 \pi - \log(\sigma_t^2) - \frac{\epsilon_t^2}{\sigma_t^2}) $$
/// Constant terms removed, since we're optimizing
pub fn neg_loglikelihood(sigma_2: &[f64], eps: &[f64]) -> f64 {
    let loglik = sigma_2.iter().zip(eps).fold(0., |acc, (sig2, ep)| {
        acc + (-sig2.ln() - (ep.powf(2.)/sig2))
    });
    -loglik
}


/// Fit the GARCH model using MLE
pub fn fit(ts: &[f64], p: usize, q: usize) -> Result<Vec<f64>, GarchError> {
    // Calculate residuals
    let mean = util::mean(ts);
    let eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();

    // Objective function
    let f = |coef: &Vec<f64>| {
        let omega = coef[0];
        let alpha = &coef[1..p+1];
        let beta = &coef[p+1..];
        let sigma_2 = garch_recursion(omega, alpha, beta, &eps);
        neg_loglikelihood(&sigma_2, &eps)
    };

    // Gradient (using automatic differentiation)
    let g = |coef: &Vec<f64>| {
        coef.forward_diff(&f)
    };

    // Really sensitive to initial values, this seems to work ok
    let mut coef: Vec<f64> = (0..1+p+q).map(|_| rand::random()).collect();
    log::debug!("Initial Guess: {:?}", coef);
    log::debug!("Initial Loss: {:?}", f(&coef));

    let n_params = coef.len();
    let mut fmin = Lbfgsb::new(&mut coef, &f, &g);
    for i in 0..n_params {
        fmin.set_lower_bound(i, 1e-8);
    }

    // For debugging
    // fmin.set_verbosity(101);
    fmin.set_verbosity(-1);
    fmin.max_iteration(100);

    let result = fmin.minimize();

    log::debug!("Final Params: {:?}", coef);
    log::debug!("Loss: {:?}", f(&coef));
    log::debug!("Opt Result: {:?}", result);
    Ok(coef)
}


/// Forecast residuals
/// GARCH gives us $\sigma^2_t$,
/// and with that we can compute $\epsilon_t$:
/// $$ \epsilon_t = \sigma_t z_t $$
/// Where $z_t$ is the white noise, which we assume to be standard normal
pub fn forecast<T: Rng>(ts: &[f64], n: usize, omega: f64, alpha: &[f64], beta: &[f64], rng: &mut T) -> Result<Vec<f64>, GarchError> {
    let mut forecaster = Forecaster::new(ts, omega, alpha, beta);

    // Forecast
    for _ in 0..n {
        forecaster.next(rng);
    }

    // Remove residuals from original time series
    forecaster.eps.drain(0..ts.len());
    Ok(forecaster.eps)
}


/// A stateful forecaster, more efficient if you
/// only need one time step ahead, e.g. for use with an ARIMA model
pub struct Forecaster {
    eps: Vec<f64>,
    sigma_2: Vec<f64>,
    noise: Normal<f64>,
    omega: f64,
    alpha: Vec<f64>,
    beta: Vec<f64>,
}

impl Forecaster {
    pub fn new(ts: &[f64], omega: f64, alpha: &[f64], beta: &[f64]) -> Forecaster {
        // Under stationarity assumptions, this mean should stay stable
        let mean = util::mean(ts);
        let eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();

        // Initialize sigma_2 for the history we have
        let sigma_2 = garch_recursion(omega, alpha, beta, &eps);

        Forecaster {
            sigma_2, eps,
            omega,
            alpha: alpha.to_owned(),
            beta: beta.to_owned(),
            noise: Normal::new(0., 1.).unwrap(),
        }
    }

    /// Returns the next predicted sigma and epsilon in the series
    pub fn next<T: Rng>(&mut self, rng: &mut T) -> (f64, f64) {
        let next_sigma_2 = predict_next(self.omega, &self.alpha, &self.beta, &self.eps, &self.sigma_2);
        self.sigma_2.push(next_sigma_2);
        let white_noise = self.noise.sample(rng);
        let next_sigma = next_sigma_2.sqrt();
        let next_eps = next_sigma * white_noise;
        self.eps.push(next_eps);
        (next_sigma, next_eps)
    }
}
