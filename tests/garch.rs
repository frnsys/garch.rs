#[cfg(test)]
mod test_garch {
    use rand::prelude::*;
    use rand_distr::{Normal, Distribution};

    fn gen_timeseries() -> Vec<f64> {
        let normal = Normal::new(0., 1.).unwrap();
        let mut rng: StdRng = SeedableRng::from_seed([100; 32]);

        // GARCH(2,2) process
        let n = 100;
        let omega = 0.5;
        let alpha = [0.1, 0.2];
        let beta = [0.3, 0.4];

        let mean = 0.;
        let mut eps: Vec<f64> = vec![normal.sample(&mut rng), normal.sample(&mut rng)];
        let mut sigma: Vec<f64> = vec![1., 1.];
        for _ in 0..n {
            let n_eps = eps.len();
            let n_sig = sigma.len();
            let sigma_next = (omega +
                              alpha[0]*eps[n_eps-1].powf(2.) + alpha[1]*eps[n_eps-2].powf(2.) +
                              beta[0]*sigma[n_sig-1].powf(2.) + beta[1]*sigma[n_sig-2].powf(2.)).sqrt();
            let eps_next = sigma_next * normal.sample(&mut rng);
            eps.push(eps_next);
            sigma.push(sigma_next);
        }
        // let sigma_2: Vec<f64> = sigma.iter().map(|s| s.powf(2.)).collect();
        // println!("sigma_2 = np.array({:?})", sigma);
        // println!("eps = np.array({:?})", eps);

        let mut ts: Vec<f64> = eps.iter().map(|e| mean + e).collect();
        ts.drain(0..2);
        ts
    }

    #[test]
    fn fit() {
        let p = 2;
        let q = 2;
        let ts = gen_timeseries();
        let coef = garch::garch::fit(&ts, p, q).unwrap();
        println!("coefficients: {:?}", coef);
    }
}

