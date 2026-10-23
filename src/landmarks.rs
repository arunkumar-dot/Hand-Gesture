//! Hand landmark model wrapper and projection of its output back to the frame.
//!
//! Ported from MediaPipe:
//! - `mediapipe/modules/hand_landmark/hand_landmark_cpu.pbtxt` (input 0..1 RGB,
//!   outputs: landmarks, presence, handedness, world landmarks)
//! - `mediapipe/calculators/tensor/tensors_to_landmarks_calculator.cc`
//!   (landmark x, y, z are in input pixels; divided by the input size)
//! - `mediapipe/calculators/util/landmark_projection_calculator.cc`

use crate::config::{LANDMARK_INPUT_SIZE, NUM_LANDMARKS};
use crate::frame::{Frame, Rect, warp_to_tensor};
use crate::palm::extract;
use crate::{Error, Result};
use ort::session::Session;
use ort::value::Tensor;

const MODEL_BYTES: &[u8] = include_bytes!("../models/hand_landmarks_detector.onnx");

/// A landmark in frame pixels; `z` is depth relative to the wrist in the same
/// units as `x` (smaller is closer to the camera).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Landmark {
    /// Horizontal position.
    pub x: f64,
    /// Vertical position.
    pub y: f64,
    /// Relative depth.
    pub z: f64,
}

impl Landmark {
    /// The origin.
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };
}

/// One landmark-model run.
#[derive(Debug, Clone)]
pub struct HandResult {
    /// The 21 landmarks in frame pixels.
    pub landmarks: [Landmark; NUM_LANDMARKS],
    /// Probability that a hand is present in the ROI.
    pub presence: f32,
    /// Raw handedness output. The frame is mirrored before inference, so
    /// "left" and "right" are swapped relative to the user's real hands. M0
    /// does not use it.
    pub handedness: f32,
    /// The ROI the model was run on.
    pub roi: Rect,
}

/// Project the model's raw landmark tensor (`21 x 3`, crop pixels) to frame
/// pixels using the inverse of the crop warp. `None` when the tensor is too short.
#[must_use]
pub fn project_landmarks(raw: &[f32], roi: &Rect) -> Option<[Landmark; NUM_LANDMARKS]> {
    let crop_to_image = roi.crop_to_image(LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE);
    let mut out = [Landmark::ZERO; NUM_LANDMARKS];
    let mut triples = raw.as_chunks::<3>().0.iter();
    for lm in &mut out {
        let &[x, y, z] = triples.next()?;
        let p = crop_to_image.apply(f64::from(x), f64::from(y));
        *lm = Landmark {
            x: p.x,
            y: p.y,
            z: f64::from(z) / f64::from(LANDMARK_INPUT_SIZE) * roi.w,
        };
    }
    Some(out)
}

/// The hand landmark model.
pub struct LandmarkModel {
    session: Session,
}

impl LandmarkModel {
    /// Load the embedded `hand_landmarks_detector.onnx`.
    ///
    /// # Errors
    /// Fails when ONNX Runtime cannot load the model.
    pub fn new() -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_memory(MODEL_BYTES)?;
        Ok(Self { session })
    }

    /// Run the model on `roi` of `frame`.
    ///
    /// # Errors
    /// Fails when inference fails or the output shapes are not the documented ones.
    pub fn run(&mut self, frame: &Frame, roi: &Rect) -> Result<HandResult> {
        let side = LANDMARK_INPUT_SIZE as usize;
        let data = warp_to_tensor(frame, roi, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE);
        let input = Tensor::from_array(([1usize, side, side, 3], data))?;
        let outputs = self.session.run(ort::inputs!["input_1" => input])?;
        let (lm_shape, raw) = extract(&outputs, "Identity")?;
        let (_, presence) = extract(&outputs, "Identity_1")?;
        let (_, handedness) = extract(&outputs, "Identity_2")?;
        let expected = i64::try_from(NUM_LANDMARKS * 3).unwrap_or(i64::MAX);
        if lm_shape != [1, expected] {
            return Err(Error::ModelOutput {
                name: "Identity",
                detail: format!("shape {lm_shape:?}, expected [1, {expected}]"),
            });
        }
        let landmarks = project_landmarks(raw, roi).ok_or_else(|| Error::ModelOutput {
            name: "Identity",
            detail: "too few values".to_owned(),
        })?;
        let scalar = |v: &[f32], name: &'static str| {
            v.first().copied().ok_or(Error::ModelOutput {
                name,
                detail: "empty".to_owned(),
            })
        };
        Ok(HandResult {
            landmarks,
            // Both scalars are already probabilities (see models/MODELS.md).
            presence: scalar(presence, "Identity_1")?,
            handedness: scalar(handedness, "Identity_2")?,
            roi: *roi,
        })
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test index arithmetic")]
mod tests {
    use super::*;

    #[test]
    fn crop_centre_projects_to_the_roi_centre() {
        let roi = Rect {
            cx: 300.0,
            cy: 200.0,
            w: 224.0,
            h: 224.0,
            rotation: 0.9,
        };
        let mut raw = vec![0.0_f32; NUM_LANDMARKS * 3];
        raw[0] = 112.0;
        raw[1] = 112.0;
        let lms = project_landmarks(&raw, &roi).expect("21 landmarks");
        assert!((lms[0].x - 300.0).abs() < 1e-4 && (lms[0].y - 200.0).abs() < 1e-4);
    }

    #[test]
    fn unrotated_roi_scales_and_offsets() {
        let roi = Rect {
            cx: 500.0,
            cy: 400.0,
            w: 448.0,
            h: 448.0,
            rotation: 0.0,
        };
        let mut raw = vec![0.0_f32; NUM_LANDMARKS * 3];
        raw[3] = 224.0; // landmark 1: bottom-right corner of the crop
        raw[4] = 224.0;
        raw[5] = 56.0; // z = 56 / 224 * 448 = 112
        let lms = project_landmarks(&raw, &roi).expect("21 landmarks");
        assert!((lms[1].x - 724.0).abs() < 1e-4 && (lms[1].y - 624.0).abs() < 1e-4);
        assert!((lms[1].z - 112.0).abs() < 1e-4);
    }

    #[test]
    fn short_tensors_are_rejected() {
        let roi = Rect {
            cx: 0.0,
            cy: 0.0,
            w: 1.0,
            h: 1.0,
            rotation: 0.0,
        };
        assert!(project_landmarks(&[0.0; 10], &roi).is_none());
    }
}
