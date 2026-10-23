//! Hand regions of interest: palm detection -> ROI and landmarks -> ROI.
//!
//! Ported from MediaPipe, with the option values read from the serialized
//! subgraph configs of the pinned `mediapipe` wheel:
//! - `palm_detection_detection_to_roi.pbtxt`: `DetectionsToRectsCalculator`
//!   (rotation from keypoint 0 to keypoint 2, target angle 90 degrees), then
//!   `RectTransformationCalculator` (scale 2.6, `shift_y` -0.5, `square_long`).
//! - `hand_landmark_landmarks_to_roi.pbtxt`: `HandLandmarksToRectCalculator`
//!   (`mediapipe/calculators/util/hand_landmarks_to_rect_calculator.cc`), then
//!   `RectTransformationCalculator` (scale 2.0, `shift_y` -0.1, `square_long`).
//! - `mediapipe/calculators/util/rect_transformation_calculator.cc`.
//!
//! Everything is in frame pixels, so MediaPipe's image-size bookkeeping
//! (normalised coordinates on non-square images) is not needed.

use crate::config::{
    LANDMARK_ROI_SCALE, LANDMARK_ROI_SHIFT_Y, NUM_LANDMARKS, PALM_ROI_SCALE, PALM_ROI_SHIFT_Y,
};
use crate::frame::Rect;
use crate::landmarks::Landmark;
use crate::palm::Detection;
use std::f64::consts::{FRAC_PI_2, PI};

/// Index of the wrist in the landmark model's topology.
const WRIST: usize = 0;
// `hand_landmarks_to_rect_calculator.cc` aligns the hand using three landmarks
// it names `*_PIP_JOINT`, although the indices (8, 12, 16) are the index,
// middle and ring fingertips of the 21-point topology: rotation points from the
// wrist to the average of the index/ring pair, averaged again with the middle one.
const INDEX_PIP: usize = 8;
const MIDDLE_PIP: usize = 12;
const RING_PIP: usize = 16;

/// Wrap an angle to `[-pi, pi)` (`NormalizeRadians`).
#[must_use]
pub fn normalize_radians(angle: f64) -> f64 {
    angle - 2.0 * PI * ((angle + PI) / (2.0 * PI)).floor()
}

/// `RectTransformationCalculator` with `square_long: true` and no x shift.
///
/// The shift is applied in the rect's rotated frame using the original width
/// and height, then the rect is squared on its long side, then scaled.
#[must_use]
pub fn transform_rect(rect: &Rect, scale: f64, shift_y: f64) -> Rect {
    let (sin, cos) = rect.rotation.sin_cos();
    let side = rect.w.max(rect.h);
    Rect {
        cx: rect.cx - rect.h * shift_y * sin,
        cy: rect.cy + rect.h * shift_y * cos,
        w: side * scale,
        h: side * scale,
        rotation: rect.rotation,
    }
}

/// Rotation that makes the vector `start -> end` point straight up on screen.
fn rotation_to_point_up(x0: f64, y0: f64, x1: f64, y1: f64) -> f64 {
    normalize_radians(FRAC_PI_2 - (-(y1 - y0)).atan2(x1 - x0))
}

/// The hand ROI for a palm detection in frame pixels.
#[must_use]
pub fn rect_from_palm(det: &Detection) -> Rect {
    let wrist = det.keypoints.first();
    let middle_base = det.keypoints.get(2);
    let rotation = match (wrist, middle_base) {
        (Some(a), Some(b)) => rotation_to_point_up(a.x, a.y, b.x, b.y),
        _ => 0.0,
    };
    let raw = Rect {
        cx: f64::midpoint(det.xmin, det.xmax),
        cy: f64::midpoint(det.ymin, det.ymax),
        w: det.xmax - det.xmin,
        h: det.ymax - det.ymin,
        rotation,
    };
    transform_rect(&raw, PALM_ROI_SCALE, PALM_ROI_SHIFT_Y)
}

/// The next frame's hand ROI from this frame's landmarks in frame pixels
/// (`HandLandmarksToRectCalculator` + `RectTransformation`).
#[must_use]
pub fn rect_from_landmarks(landmarks: &[Landmark; NUM_LANDMARKS]) -> Rect {
    let at = |i: usize| landmarks.get(i).copied().unwrap_or(Landmark::ZERO);
    let wrist = at(WRIST);
    let x1 = f64::midpoint(
        f64::midpoint(at(INDEX_PIP).x, at(RING_PIP).x),
        at(MIDDLE_PIP).x,
    );
    let y1 = f64::midpoint(
        f64::midpoint(at(INDEX_PIP).y, at(RING_PIP).y),
        at(MIDDLE_PIP).y,
    );
    let rotation = rotation_to_point_up(wrist.x, wrist.y, x1, y1);

    // Axis-aligned centre of all landmarks.
    let (mut min_x, mut max_x) = (f64::MAX, f64::MIN);
    let (mut min_y, mut max_y) = (f64::MAX, f64::MIN);
    for lm in landmarks {
        min_x = min_x.min(lm.x);
        max_x = max_x.max(lm.x);
        min_y = min_y.min(lm.y);
        max_y = max_y.max(lm.y);
    }
    let center_x = f64::midpoint(min_x, max_x);
    let center_y = f64::midpoint(min_y, max_y);

    // Bounds of the landmarks in the hand's own (de-rotated) frame.
    let (sin, cos) = normalize_radians(-rotation).sin_cos();
    let (mut pmin_x, mut pmax_x) = (f64::MAX, f64::MIN);
    let (mut pmin_y, mut pmax_y) = (f64::MAX, f64::MIN);
    for lm in landmarks {
        let ox = lm.x - center_x;
        let oy = lm.y - center_y;
        let px = ox * cos - oy * sin;
        let py = ox * sin + oy * cos;
        pmin_x = pmin_x.min(px);
        pmax_x = pmax_x.max(px);
        pmin_y = pmin_y.min(py);
        pmax_y = pmax_y.max(py);
    }
    let pcx = f64::midpoint(pmin_x, pmax_x);
    let pcy = f64::midpoint(pmin_y, pmax_y);
    let (rsin, rcos) = rotation.sin_cos();
    let raw = Rect {
        cx: pcx * rcos - pcy * rsin + center_x,
        cy: pcx * rsin + pcy * rcos + center_y,
        w: pmax_x - pmin_x,
        h: pmax_y - pmin_y,
        rotation,
    };
    transform_rect(&raw, LANDMARK_ROI_SCALE, LANDMARK_ROI_SHIFT_Y)
}

/// IoU of the axis-aligned bounds of two ROIs (`AssociationNormRectCalculator`
/// compares rects without regard to rotation).
#[must_use]
pub fn rect_iou(a: &Rect, b: &Rect) -> f64 {
    let (ax0, ay0, ax1, ay1) = a.bounds();
    let (bx0, by0, bx1, by1) = b.bounds();
    let iw = (ax1.min(bx1) - ax0.max(bx0)).max(0.0);
    let ih = (ay1.min(by1) - ay0.max(by0)).max(0.0);
    let inter = iw * ih;
    let union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test index arithmetic")]
mod tests {
    use super::*;
    use crate::config::NUM_PALM_KEYPOINTS;
    use crate::frame::Point;

    fn palm(wrist: (f64, f64), middle: (f64, f64), bbox: (f64, f64, f64, f64)) -> Detection {
        let mut keypoints = [Point { x: 0.0, y: 0.0 }; NUM_PALM_KEYPOINTS];
        keypoints[0] = Point {
            x: wrist.0,
            y: wrist.1,
        };
        keypoints[2] = Point {
            x: middle.0,
            y: middle.1,
        };
        Detection {
            score: 0.9,
            xmin: bbox.0,
            ymin: bbox.1,
            xmax: bbox.2,
            ymax: bbox.3,
            keypoints,
        }
    }

    #[test]
    fn upright_palm_has_no_rotation_and_the_expected_box() {
        // Wrist below the middle-finger base: hand points up.
        let det = palm((100.0, 200.0), (100.0, 100.0), (80.0, 130.0, 120.0, 170.0));
        let r = rect_from_palm(&det);
        assert!(r.rotation.abs() < 1e-12);
        // Shift -0.5 * 40 = -20 (up), side 40 * 2.6 = 104.
        assert!((r.cx - 100.0).abs() < 1e-9 && (r.cy - 130.0).abs() < 1e-9);
        assert!((r.w - 104.0).abs() < 1e-9 && (r.h - 104.0).abs() < 1e-9);
    }

    #[test]
    fn palm_pointing_right_rotates_a_quarter_turn_and_shifts_along_the_hand() {
        let det = palm((0.0, 100.0), (50.0, 100.0), (-20.0, 80.0, 20.0, 120.0));
        let r = rect_from_palm(&det);
        assert!((r.rotation - FRAC_PI_2).abs() < 1e-9);
        // "Up" for this hand is screen-right: the centre moves +20 in x.
        assert!((r.cx - 20.0).abs() < 1e-9 && (r.cy - 100.0).abs() < 1e-9);
    }

    #[test]
    fn palm_tilted_clockwise_gets_a_positive_rotation() {
        let theta: f64 = 0.4;
        let tip = (100.0 + 100.0 * theta.sin(), 200.0 - 100.0 * theta.cos());
        let r = rect_from_palm(&palm((100.0, 200.0), tip, (80.0, 130.0, 120.0, 170.0)));
        assert!((r.rotation - theta).abs() < 1e-9);
    }

    #[test]
    fn palm_roi_uses_the_long_side_of_a_non_square_box() {
        let det = palm((0.0, 100.0), (0.0, 0.0), (-10.0, 40.0, 10.0, 100.0));
        let r = rect_from_palm(&det);
        assert!((r.w - 60.0 * 2.6).abs() < 1e-9);
        // Shift uses the original height (60): centre y 70 - 0.5 * 60 = 40.
        assert!((r.cy - 40.0).abs() < 1e-9);
    }

    #[test]
    fn normalize_radians_wraps_into_range() {
        assert!((normalize_radians(3.0 * PI) - (-PI)).abs() < 1e-9);
        assert!((normalize_radians(-0.5) - (-0.5)).abs() < 1e-12);
        assert!((normalize_radians(2.0 * PI + 0.25) - 0.25).abs() < 1e-9);
    }

    fn hand_along(theta: f64, length: f64, wrist: (f64, f64)) -> [Landmark; NUM_LANDMARKS] {
        let mut out = [Landmark::ZERO; NUM_LANDMARKS];
        for (i, lm) in out.iter_mut().enumerate() {
            let d = length * i as f64 / (NUM_LANDMARKS - 1) as f64;
            *lm = Landmark {
                x: wrist.0 + d * theta.sin(),
                y: wrist.1 - d * theta.cos(),
                z: 0.0,
            };
        }
        out
    }

    #[test]
    fn upright_landmarks_give_the_expected_next_roi() {
        let r = rect_from_landmarks(&hand_along(0.0, 200.0, (100.0, 300.0)));
        assert!(r.rotation.abs() < 1e-9);
        // Raw rect: centre (100, 200), 0 x 200. Shift -0.1 * 200 = -20, side 200, scale 2.
        assert!((r.cx - 100.0).abs() < 1e-9 && (r.cy - 180.0).abs() < 1e-9);
        assert!((r.w - 400.0).abs() < 1e-9 && (r.h - 400.0).abs() < 1e-9);
    }

    #[test]
    fn tilted_landmarks_give_a_tilted_roi_centred_on_the_hand() {
        let theta = 0.5;
        let r = rect_from_landmarks(&hand_along(theta, 200.0, (100.0, 300.0)));
        assert!((r.rotation - theta).abs() < 1e-9);
        // Raw centre is the middle of the segment; the shift moves it 20 px up the hand axis.
        let mid = (100.0 + 100.0 * theta.sin(), 300.0 - 100.0 * theta.cos());
        let expected = (mid.0 + 20.0 * theta.sin(), mid.1 - 20.0 * theta.cos());
        assert!((r.cx - expected.0).abs() < 1e-9 && (r.cy - expected.1).abs() < 1e-9);
        assert!((r.w - 400.0).abs() < 1e-9);
    }

    #[test]
    fn rect_iou_of_identical_and_disjoint_rects() {
        let a = Rect {
            cx: 0.0,
            cy: 0.0,
            w: 10.0,
            h: 10.0,
            rotation: 0.0,
        };
        let far = Rect { cx: 100.0, ..a };
        let half = Rect { cx: 5.0, ..a };
        assert!((rect_iou(&a, &a) - 1.0).abs() < 1e-12);
        assert!(rect_iou(&a, &far).abs() < 1e-12);
        assert!((rect_iou(&a, &half) - 1.0 / 3.0).abs() < 1e-12);
    }
}
