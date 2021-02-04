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
    // Really sensitive to initial values, this seems to work ok
    let mut coef: Vec<f64> = (0..1+p+q).map(|_| rand::random()).collect();
    println!("initial guess: {:?}", coef);

    // Calculate residuals
    let mean = util::mean(ts);
    let eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();

    let f = |coef: &Vec<f64>| {
        let omega = coef[0];
        let alpha = &coef[1..p+1];
        let beta = &coef[p+1..];
        let sigma_2 = garch_recursion(omega, alpha, beta, &eps);
        neg_loglikelihood(&sigma_2, &eps)
    };

    let g = |coef: &Vec<f64>| {
        coef.forward_diff(&f)
    };
    let n_params = coef.len();
    let mut fmin = Lbfgsb::new(&mut coef, &f, &g);
    for i in 0..n_params {
        fmin.set_lower_bound(i, 1e-8);
    }

    // For debugging
    // fmin.set_verbosity(101);

    fmin.minimize();

    Ok(coef)
}


/// Forecast residuals
/// GARCH gives us $\sigma^2_t$,
/// and with that we can compute $\epsilon_t$:
/// $$ \epsilon_t = \sigma_t z_t $$
/// Where $z_t$ is the white noise, which we assume to be standard normal
pub fn forecast<T: Rng>(ts: &[f64], n: usize, omega: f64, alpha: &[f64], beta: &[f64], rng: &mut T) -> Result<Vec<f64>, GarchError> {
    let mean = util::mean(ts);
    let mut eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();

    // White noise function
    let normal = Normal::new(0., 1.).unwrap();

    // Initialize sigma_2 for the history we have
    let mut sigma_2 = garch_recursion(omega, alpha, beta, &eps);

    // Forecast
    for _ in 0..n {
        let next_sigma_2 = predict_next(omega, alpha, beta, &eps, &sigma_2);
        sigma_2.push(next_sigma_2);
        let white_noise = normal.sample(rng);
        let next_eps = next_sigma_2.sqrt() * white_noise;
        eps.push(next_eps);
    }

    // Remove residuals from original time series
    eps.drain(0..ts.len());
    Ok(eps)
}
