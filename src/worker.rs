//! The worker thread: camera capture and inference.
//!
//! It publishes the newest [`Snapshot`] to a [`LatestSlot`]; the render loop
//! only ever takes the latest value, so a slow inference never blocks drawing
//! and stale frames are dropped.

use crate::Result;
use crate::camera::Camera;
use crate::config::{FilterConfig, JITTER_TEST_SECONDS, PALM_CENTER_LANDMARKS};
use crate::filter::rms_deviation;
use crate::frame::Frame;
use crate::landmarks::Landmark;
use crate::tracker::{Mode, TrackedHand, Tracker};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::Instant;

/// Result of a finished jitter test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JitterReport {
    /// RMS deviation of the raw palm centre from its mean, in pixels.
    pub raw_rms: f64,
    /// RMS deviation of the filtered palm centre from its mean, in pixels.
    pub filtered_rms: f64,
    /// Number of frames sampled.
    pub samples: usize,
}

/// State of the jitter test, for the HUD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum JitterStatus {
    /// Not running and never run.
    Idle,
    /// Collecting; seconds left.
    Recording(f64),
    /// Finished with a result.
    Done(JitterReport),
    /// Finished but too few frames had a hand.
    NoHand,
}

/// Everything the renderer needs about one processed frame.
#[derive(Debug, Clone)]
pub struct Snapshot {
    /// The mirrored frame that was processed.
    pub frame: Arc<Frame>,
    /// Hands found in it.
    pub hands: Vec<TrackedHand>,
    /// Whether palm detection ran.
    pub mode: Mode,
    /// Palm detection time, when it ran.
    pub palm_ms: Option<f64>,
    /// Mean landmark-model time over this frame's runs, when any ran.
    pub landmark_ms: Option<f64>,
    /// Frames per second read from the camera.
    pub camera_fps: f64,
    /// Frames per second through inference.
    pub inference_fps: f64,
    /// Jitter-test state.
    pub jitter: JitterStatus,
}

/// A single-value mailbox: publishing replaces whatever was there.
#[derive(Debug)]
pub struct LatestSlot<T> {
    value: Mutex<Option<T>>,
}

impl<T> Default for LatestSlot<T> {
    fn default() -> Self {
        Self {
            value: Mutex::new(None),
        }
    }
}

impl<T> LatestSlot<T> {
    fn lock(&self) -> MutexGuard<'_, Option<T>> {
        // A poisoned slot still holds a valid value; keep going.
        self.value.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Replace the stored value, dropping an unread older one.
    pub fn publish(&self, value: T) {
        *self.lock() = Some(value);
    }

    /// Take the stored value, if a new one arrived since the last take.
    pub fn take(&self) -> Option<T> {
        self.lock().take()
    }
}

/// UI -> worker controls and worker -> UI fatal error.
#[derive(Debug)]
pub struct Controls {
    /// Display the filtered landmarks (the filter itself always runs).
    pub filter_on: AtomicBool,
    /// Set by the UI to start a jitter test.
    pub jitter_request: AtomicBool,
    /// Set by the UI to stop the worker.
    pub quit: AtomicBool,
    /// Set by the worker when it dies.
    pub error: Mutex<Option<String>>,
}

impl Controls {
    /// Controls with the filter on.
    #[must_use]
    pub fn new() -> Self {
        Self {
            filter_on: AtomicBool::new(true),
            jitter_request: AtomicBool::new(false),
            quit: AtomicBool::new(false),
            error: Mutex::new(None),
        }
    }

    /// The worker's fatal error, if any.
    #[must_use]
    pub fn error_message(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

impl Default for Controls {
    fn default() -> Self {
        Self::new()
    }
}

/// Frames-per-second over a sliding one-second window.
struct FpsCounter {
    window_start: Instant,
    count: u32,
    fps: f64,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            window_start: Instant::now(),
            count: 0,
            fps: 0.0,
        }
    }

    fn tick(&mut self) -> f64 {
        self.count += 1;
        let dt = self.window_start.elapsed().as_secs_f64();
        if dt >= 1.0 {
            self.fps = f64::from(self.count) / dt;
            self.count = 0;
            self.window_start = Instant::now();
        }
        self.fps
    }
}

/// Palm centre: average of landmarks 0, 5, 9, 13 and 17, in pixels.
#[must_use]
pub fn palm_center(landmarks: &[Landmark]) -> (f64, f64) {
    let mut sum = (0.0, 0.0);
    let mut n = 0.0;
    for &i in &PALM_CENTER_LANDMARKS {
        if let Some(lm) = landmarks.get(i) {
            sum.0 += lm.x;
            sum.1 += lm.y;
            n += 1.0;
        }
    }
    if n > 0.0 {
        (sum.0 / n, sum.1 / n)
    } else {
        (0.0, 0.0)
    }
}

/// The 3-second hold-still test: samples the palm centre of the first hand.
#[derive(Debug, Default)]
struct JitterTest {
    started: Option<f64>,
    raw: Vec<(f64, f64)>,
    filtered: Vec<(f64, f64)>,
}

/// Fewer samples than this make the RMS meaningless.
const MIN_JITTER_SAMPLES: usize = 10;

impl JitterTest {
    fn start(&mut self, t: f64) {
        self.started = Some(t);
        self.raw.clear();
        self.filtered.clear();
    }

    /// Feed one frame; returns the status to display, or `None` when idle.
    fn update(&mut self, hands: &[TrackedHand], t: f64) -> Option<JitterStatus> {
        let started = self.started?;
        if let Some(hand) = hands.first() {
            self.raw.push(palm_center(&hand.raw));
            self.filtered.push(palm_center(&hand.smoothed));
        }
        let elapsed = t - started;
        if elapsed < JITTER_TEST_SECONDS {
            return Some(JitterStatus::Recording(JITTER_TEST_SECONDS - elapsed));
        }
        self.started = None;
        if self.raw.len() < MIN_JITTER_SAMPLES {
            println!("jitter test: no hand tracked, try again");
            return Some(JitterStatus::NoHand);
        }
        let report = JitterReport {
            raw_rms: rms_deviation(&self.raw),
            filtered_rms: rms_deviation(&self.filtered),
            samples: self.raw.len(),
        };
        println!(
            "jitter test ({} frames): raw RMS {:.3} px, filtered RMS {:.3} px",
            report.samples, report.raw_rms, report.filtered_rms
        );
        Some(JitterStatus::Done(report))
    }
}

fn run(controls: &Controls, slot: &LatestSlot<Snapshot>) -> Result<()> {
    let mut camera = Camera::open()?;
    let mut tracker = Tracker::new(FilterConfig::default())?;
    let origin = Instant::now();
    let mut camera_fps = FpsCounter::new();
    let mut inference_fps = FpsCounter::new();
    let mut jitter = JitterTest::default();
    let mut jitter_status = JitterStatus::Idle;

    while !controls.quit.load(Ordering::Relaxed) {
        // Mirror once, before inference and display, so drawn points match the image.
        let frame = Arc::new(camera.next_frame()?.mirrored());
        let cam = camera_fps.tick();
        let t = origin.elapsed().as_secs_f64();

        if controls.jitter_request.swap(false, Ordering::Relaxed) {
            jitter.start(t);
        }
        let result = tracker.process(&frame, t)?;
        let inf = inference_fps.tick();
        if let Some(status) = jitter.update(&result.hands, t) {
            jitter_status = status;
        }

        let landmark_ms = if result.landmark_ms.is_empty() {
            None
        } else {
            #[allow(clippy::cast_precision_loss, reason = "at most a few runs per frame")]
            let n = result.landmark_ms.len() as f64;
            Some(result.landmark_ms.iter().sum::<f64>() / n)
        };
        slot.publish(Snapshot {
            frame,
            hands: result.hands,
            mode: result.mode,
            palm_ms: result.palm_ms,
            landmark_ms,
            camera_fps: cam,
            inference_fps: inf,
            jitter: jitter_status,
        });
    }
    Ok(())
}

/// Start the worker thread. Errors end the thread and are stored in `controls`.
///
/// # Errors
/// Fails when the thread cannot be spawned.
pub fn spawn(
    controls: Arc<Controls>,
    slot: Arc<LatestSlot<Snapshot>>,
) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new()
        .name("handwave-worker".to_owned())
        .spawn(move || {
            if let Err(e) = run(&controls, &slot) {
                *controls
                    .error
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner) = Some(e.to_string());
            }
        })
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test index arithmetic")]
mod tests {
    use super::*;

    #[test]
    fn slot_keeps_only_the_latest_value() {
        let slot = LatestSlot::default();
        slot.publish(1);
        slot.publish(2);
        assert_eq!(slot.take(), Some(2));
        assert_eq!(slot.take(), None);
    }

    #[test]
    fn palm_center_averages_the_five_palm_landmarks() {
        let mut lms = [Landmark::ZERO; 21];
        for (i, lm) in lms.iter_mut().enumerate() {
            lm.x = i as f64;
            lm.y = 2.0 * i as f64;
        }
        // (0 + 5 + 9 + 13 + 17) / 5 = 8.8
        let c = palm_center(&lms);
        assert!((c.0 - 8.8).abs() < 1e-12 && (c.1 - 17.6).abs() < 1e-12);
    }
}
