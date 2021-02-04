use rand::Rng;
use super::util;
use super::error::GarchError;
use rand_distr::{Normal, Distribution};
use argmin::prelude::*;
use argmin::solver::linesearch::MoreThuenteLineSearch;
use argmin::solver::quasinewton::LBFGS;

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

fn neg_loglikelihood_grad(omega: f64, alpha: &[f64], beta: &[f64], eps: &[f64]) -> Vec<f64> {
    let grad = vec![0.; 1 + alpha.len() + beta.len()];
    let sigma_2 = garch_recursion(omega, alpha, beta, eps);
    sigma_2.iter().zip(eps).fold(grad, |mut acc, (sig2, ep)| {
        let r = ep.powf(2.)/sig2.powf(2.);
        acc[0] += -(1./sig2 + r);
        for i in 0..alpha.len() {
            acc[1+i] += -(alpha[i]/sig2 + alpha[i]*r);
        }
        for i in 0..beta.len() {
            acc[1+alpha.len()+i] += -(beta[i]/sig2 + beta[i]*r);
        }
        acc
    })
}


#[derive(Clone, Default)]
struct FitGARCH {
    p: usize,
    eps: Vec<f64>,
}
impl ArgminOp for FitGARCH {
    type Param = Vec<f64>;
    type Output = f64;
    type Hessian = ();
    type Jacobian = ();
    type Float = f64;

    /// Apply the cost function to a parameter `p`
    fn apply(&self, coef: &Self::Param) -> Result<Self::Output, Error> {
        let omega = coef[0];
        let alpha = &coef[1..self.p+1];
        let beta = &coef[self.p+1..];
        let sigma_2 = garch_recursion(omega, alpha, beta, &self.eps);
        let nll = neg_loglikelihood(&sigma_2, &self.eps);
        Ok(nll)
    }

    /// Compute the gradient at parameter `p`.
    fn gradient(&self, coef: &Self::Param) -> Result<Self::Param, Error> {
        let omega = coef[0];
        let alpha = &coef[1..self.p+1];
        let beta = &coef[self.p+1..];
        let grad = neg_loglikelihood_grad(omega, alpha, beta, &self.eps);
        Ok(grad)
    }
}

/// Fit the GARCH model using MLE
pub fn fit(ts: &[f64], p: usize, q: usize) -> Result<Vec<f64>, GarchError> {
    // Really sensitive to initial values, this seems to work ok
    let coef = (0..1+p+q).map(|_| rand::random()).collect();
    println!("initial guess: {:?}", coef);

    // Calculate residuals
    let mean = util::mean(ts);
    let eps: Vec<f64> = ts.iter().map(|x| x - mean).collect();

    // Set up solver
    let linesearch = MoreThuenteLineSearch::new().c(1e-4, 0.9).unwrap();
    let solver = LBFGS::new(linesearch, 7);
        // .with_tol_grad(1e-8)
        // .with_tol_cost(1e-8);
    let cost = FitGARCH {
        p, eps
    };

    let res = Executor::new(cost, solver, coef)
        .add_observer(ArgminSlogLogger::term(), ObserverMode::Always)
        .max_iters(100)
        .run().unwrap();

    Ok(res.state.param)
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
