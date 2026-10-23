//! One Euro filter (Casiez, Roussel, Vogel: "1 Euro Filter: A Simple
//! Speed-based Low-pass Filter for Noisy Input in Interactive Systems", CHI 2012).
//!
//! Pure: time is passed in by the caller, so the filter is deterministic.

use crate::config::{FilterConfig, NUM_LANDMARKS};
use crate::landmarks::Landmark;
use std::f64::consts::PI;

/// One Euro filter for a single scalar signal.
#[derive(Debug, Clone)]
pub struct OneEuroFilter {
    config: FilterConfig,
    state: Option<State>,
}

#[derive(Debug, Clone, Copy)]
struct State {
    x: f64,
    dx: f64,
    t: f64,
}

fn smoothing_factor(dt: f64, cutoff: f64) -> f64 {
    let tau = 1.0 / (2.0 * PI * cutoff);
    1.0 / (1.0 + tau / dt)
}

impl OneEuroFilter {
    /// A filter with no history.
    #[must_use]
    pub fn new(config: FilterConfig) -> Self {
        Self {
            config,
            state: None,
        }
    }

    /// Forget all history; the next sample passes through unchanged.
    pub fn reset(&mut self) {
        self.state = None;
    }

    /// Filter one sample taken at time `t` (seconds).
    pub fn filter(&mut self, x: f64, t: f64) -> f64 {
        let Some(prev) = self.state else {
            self.state = Some(State { x, dx: 0.0, t });
            return x;
        };
        let dt = t - prev.t;
        if dt <= f64::EPSILON {
            // Same or earlier timestamp: no new information about speed.
            return prev.x;
        }
        let raw_dx = (x - prev.x) / dt;
        let a_d = smoothing_factor(dt, self.config.d_cutoff);
        let dx = a_d * raw_dx + (1.0 - a_d) * prev.dx;
        let cutoff = self.config.min_cutoff + self.config.beta * dx.abs();
        let a = smoothing_factor(dt, cutoff);
        let x_hat = a * x + (1.0 - a) * prev.x;
        self.state = Some(State { x: x_hat, dx, t });
        x_hat
    }
}

/// One filter per coordinate of the 21 landmarks of a track.
#[derive(Debug, Clone)]
pub struct LandmarkFilter {
    filters: Vec<[OneEuroFilter; 3]>,
}

impl LandmarkFilter {
    /// A fresh filter bank (a track just started).
    #[must_use]
    pub fn new(config: FilterConfig) -> Self {
        let axis = || OneEuroFilter::new(config);
        Self {
            filters: (0..NUM_LANDMARKS)
                .map(|_| [axis(), axis(), axis()])
                .collect(),
        }
    }

    /// Reset every coordinate filter.
    pub fn reset(&mut self) {
        for axes in &mut self.filters {
            for f in axes {
                f.reset();
            }
        }
    }

    /// Filter a full hand sampled at time `t` (seconds).
    pub fn apply(
        &mut self,
        landmarks: &[Landmark; NUM_LANDMARKS],
        t: f64,
    ) -> [Landmark; NUM_LANDMARKS] {
        let mut out = *landmarks;
        for (lm, axes) in out.iter_mut().zip(&mut self.filters) {
            let [fx, fy, fz] = axes;
            lm.x = fx.filter(lm.x, t);
            lm.y = fy.filter(lm.y, t);
            lm.z = fz.filter(lm.z, t);
        }
        out
    }
}

/// Root-mean-square distance of `points` from their mean.
#[must_use]
pub fn rms_deviation(points: &[(f64, f64)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    #[allow(clippy::cast_precision_loss, reason = "sample counts are tiny")]
    let n = points.len() as f64;
    let mx = points.iter().map(|p| p.0).sum::<f64>() / n;
    let my = points.iter().map(|p| p.1).sum::<f64>() / n;
    let sum_sq: f64 = points
        .iter()
        .map(|p| (p.0 - mx).powi(2) + (p.1 - my).powi(2))
        .sum();
    (sum_sq / n).sqrt()
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test index arithmetic")]
mod tests {
    use super::*;

    /// Deterministic pseudo-noise in `[-0.5, 0.5)` (no rand dependency).
    fn noise(i: u32) -> f64 {
        let mut x = i.wrapping_mul(2_654_435_761).wrapping_add(12_345);
        x ^= x >> 15;
        x = x.wrapping_mul(2_246_822_519);
        x ^= x >> 13;
        f64::from(x % 10_000) / 10_000.0 - 0.5
    }

    fn variance(v: &[f64]) -> f64 {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64
    }

    #[test]
    fn first_sample_passes_through() {
        let mut f = OneEuroFilter::new(FilterConfig::default());
        assert!((f.filter(42.0, 0.0) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn constant_input_converges_to_the_constant() {
        let mut f = OneEuroFilter::new(FilterConfig::default());
        f.filter(0.0, 0.0);
        let mut out = 0.0;
        for i in 1..=300 {
            out = f.filter(10.0, f64::from(i) / 30.0);
        }
        assert!((out - 10.0).abs() < 1e-3, "got {out}");
    }

    #[test]
    fn noisy_constant_has_lower_variance_than_the_input() {
        let mut f = OneEuroFilter::new(FilterConfig::default());
        let input: Vec<f64> = (0..300).map(|i| 100.0 + 2.0 * noise(i)).collect();
        let output: Vec<f64> = input
            .iter()
            .enumerate()
            .map(|(i, x)| f.filter(*x, i as f64 / 30.0))
            .collect();
        assert!(variance(&output) < 0.5 * variance(&input));
    }

    #[test]
    fn follows_fast_motion_closely() {
        let mut f = OneEuroFilter::new(FilterConfig::default());
        let mut out = 0.0;
        for i in 0..90 {
            out = f.filter(f64::from(i) * 20.0, f64::from(i) / 30.0);
        }
        // 600 px/s: the speed term opens the cutoff, so lag stays small.
        assert!(
            (f64::from(89) * 20.0 - out).abs() < 60.0,
            "lag too large: {out}"
        );
    }

    #[test]
    fn reset_forgets_history() {
        let mut f = OneEuroFilter::new(FilterConfig::default());
        f.filter(0.0, 0.0);
        f.filter(0.0, 0.1);
        f.reset();
        assert!((f.filter(500.0, 0.2) - 500.0).abs() < 1e-12);
    }

    #[test]
    fn repeated_timestamp_returns_previous_output() {
        let mut f = OneEuroFilter::new(FilterConfig::default());
        f.filter(1.0, 1.0);
        assert!((f.filter(99.0, 1.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rms_deviation_of_known_points() {
        assert!(rms_deviation(&[]).abs() < 1e-12);
        assert!(rms_deviation(&[(1.0, 1.0), (1.0, 1.0)]).abs() < 1e-12);
        // Points at +-1 on x: mean 0, RMS 1.
        assert!((rms_deviation(&[(1.0, 0.0), (-1.0, 0.0)]) - 1.0).abs() < 1e-12);
    }
}
