pub mod util;
pub mod garch;
pub mod error;

pub use crate::garch::{fit, forecast};
