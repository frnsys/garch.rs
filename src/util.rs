/// Calculate the mean of a vector
pub fn mean(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    x.iter().sum::<f64>() / n
}

/// Calculate the std of a vector
pub fn std(x: &[f64]) -> f64 {
    let u = mean(x);
    let sqdevs: Vec<f64> = x.iter().map(|v| (v - u).powf(2.)).collect();
    mean(&sqdevs).sqrt()
}

