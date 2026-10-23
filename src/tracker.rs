//! Detect-versus-track state machine for up to two hands.
//!
//! Follows `mediapipe/modules/hand_landmark/hand_landmark_tracking_cpu.pbtxt`:
//! tracked hands get their next ROI from the previous landmarks and skip palm
//! detection; the palm detector only runs while fewer than [`MAX_HANDS`] hands
//! are tracked, and detections that overlap an existing ROI are dropped
//! (`AssociationNormRectCalculator`, IoU > [`DUPLICATE_IOU`]).

use crate::Result;
use crate::config::{DUPLICATE_IOU, FilterConfig, MAX_HANDS, NUM_LANDMARKS, PRESENCE_THRESHOLD};
use crate::filter::LandmarkFilter;
use crate::frame::{Frame, Rect};
use crate::landmarks::{HandResult, Landmark, LandmarkModel};
use crate::palm::PalmDetector;
use crate::roi::{rect_from_landmarks, rect_from_palm, rect_iou};
use std::time::Instant;

/// Whether the palm detector ran for a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The palm detector ran (fewer than two hands were tracked).
    Detecting,
    /// Every hand came from the previous frame's landmarks.
    Tracking,
}

impl Mode {
    /// Lowercase label for the HUD.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Detecting => "detecting",
            Self::Tracking => "tracking",
        }
    }
}

/// One tracked hand in one frame.
#[derive(Debug, Clone)]
pub struct TrackedHand {
    /// Stable id of the track.
    pub id: u64,
    /// Landmarks straight from the model, in frame pixels.
    pub raw: [Landmark; NUM_LANDMARKS],
    /// Landmarks after the One Euro filter (always computed; the UI picks).
    pub smoothed: [Landmark; NUM_LANDMARKS],
    /// The ROI the model was run on.
    pub roi: Rect,
    /// Hand presence probability.
    pub presence: f32,
    /// Raw handedness output (mirrored frame; unused in M0).
    pub handedness: f32,
}

/// Output of [`Tracker::process`].
#[derive(Debug, Clone)]
pub struct FrameResult {
    /// Hands found in this frame.
    pub hands: Vec<TrackedHand>,
    /// Whether palm detection ran.
    pub mode: Mode,
    /// Palm detection time, when it ran.
    pub palm_ms: Option<f64>,
    /// Time of every landmark-model run this frame.
    pub landmark_ms: Vec<f64>,
}

struct Track {
    id: u64,
    /// ROI to use on the next frame.
    next_roi: Rect,
    filter: LandmarkFilter,
}

/// The tracker: owns both models and the per-hand state.
pub struct Tracker {
    palm: PalmDetector,
    landmarks: LandmarkModel,
    tracks: Vec<Track>,
    filter_config: FilterConfig,
    next_id: u64,
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

impl Tracker {
    /// Load both models.
    ///
    /// # Errors
    /// Fails when a model cannot be loaded.
    pub fn new(filter_config: FilterConfig) -> Result<Self> {
        Ok(Self {
            palm: PalmDetector::new()?,
            landmarks: LandmarkModel::new()?,
            tracks: Vec::new(),
            filter_config,
            next_id: 0,
        })
    }

    /// Drop every track (the next frame detects from scratch); models stay loaded.
    pub fn reset(&mut self) {
        self.tracks.clear();
    }

    /// Number of hands currently tracked.
    #[must_use]
    pub fn tracked(&self) -> usize {
        self.tracks.len()
    }

    /// Process one frame captured at time `t` (seconds, monotonic).
    ///
    /// # Errors
    /// Fails when inference fails.
    pub fn process(&mut self, frame: &Frame, t: f64) -> Result<FrameResult> {
        let mut hands = Vec::new();
        let mut landmark_ms = Vec::new();
        let mut used_rois: Vec<Rect> = Vec::new();
        let mut survivors = Vec::new();

        // 1. Tracked hands: ROI from the previous landmarks, no palm detection.
        for mut track in std::mem::take(&mut self.tracks) {
            let start = Instant::now();
            let result = self.landmarks.run(frame, &track.next_roi)?;
            landmark_ms.push(elapsed_ms(start));
            used_rois.push(track.next_roi);
            if result.presence < PRESENCE_THRESHOLD {
                continue; // hand lost
            }
            track.next_roi = rect_from_landmarks(&result.landmarks);
            used_rois.push(track.next_roi);
            hands.push(hand_from_mut(&mut track, &result, t));
            survivors.push(track);
        }
        self.tracks = survivors;

        // 2. Fewer than MAX_HANDS tracked: look for new palms.
        let mut mode = Mode::Tracking;
        let mut palm_ms = None;
        if self.tracks.len() < MAX_HANDS {
            mode = Mode::Detecting;
            let start = Instant::now();
            let detections = self.palm.detect(frame)?;
            palm_ms = Some(elapsed_ms(start));
            for det in &detections {
                if self.tracks.len() >= MAX_HANDS {
                    break;
                }
                let roi = rect_from_palm(det);
                if used_rois.iter().any(|r| rect_iou(r, &roi) > DUPLICATE_IOU) {
                    continue;
                }
                let start = Instant::now();
                let result = self.landmarks.run(frame, &roi)?;
                landmark_ms.push(elapsed_ms(start));
                used_rois.push(roi);
                if result.presence < PRESENCE_THRESHOLD {
                    continue;
                }
                let mut track = Track {
                    id: self.next_id,
                    next_roi: rect_from_landmarks(&result.landmarks),
                    // A new track starts with a fresh filter (reset on start).
                    filter: LandmarkFilter::new(self.filter_config),
                };
                self.next_id += 1;
                used_rois.push(track.next_roi);
                hands.push(hand_from_mut(&mut track, &result, t));
                self.tracks.push(track);
            }
        }

        Ok(FrameResult {
            hands,
            mode,
            palm_ms,
            landmark_ms,
        })
    }
}

fn hand_from_mut(track: &mut Track, result: &HandResult, t: f64) -> TrackedHand {
    TrackedHand {
        id: track.id,
        raw: result.landmarks,
        smoothed: track.filter.apply(&result.landmarks, t),
        roi: result.roi,
        presence: result.presence,
        handedness: result.handedness,
    }
}
