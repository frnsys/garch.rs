#[cfg(test)]
mod test_garch {
    use rand::prelude::*;
    use rand_distr::{Normal, Distribution};

    fn init() {
        let _ = env_logger::builder().is_test(true).try_init();
    }

    fn gen_timeseries(n: usize) -> Vec<f64> {
        let normal = Normal::new(0., 1.).unwrap();
        let mut rng: StdRng = SeedableRng::from_seed([100; 32]);

        // GARCH(2,2) process
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
        init();
        let p = 2;
        let q = 2;
        let ts = gen_timeseries(100);
        let coef = garch::fit(&ts, p, q).unwrap();
        log::debug!("Coefficients: {:?}", coef);
    }

    #[test]
    fn forecast() {
        init();
        let mut rng: StdRng = SeedableRng::from_seed([100; 32]);
        let ts = gen_timeseries(100);
        let omega = 0.5;
        let alpha = [0.1, 0.2];
        let beta = [0.3, 0.4];
        let forecast = garch::forecast(
            &ts,
            100,
            omega,
            &alpha,
            &beta,
            &mut rng
        ).unwrap();
        assert_eq!(forecast.len(), 100);

        let mean = garch::util::mean(&ts);
        let std = garch::util::std(&ts);
        log::debug!("Initial  Mean:{:?} Std:{:?}", mean, std);

        let mean_ = garch::util::mean(&forecast);
        let std_ = garch::util::std(&forecast);
        log::debug!("Forecast Mean:{:?} Std:{:?}", mean_, std_);

        assert!((mean - mean_).abs() < 0.05);
        assert!((std - std_).abs() < 0.1);
    }
}

