//! Palm detector: SSD anchors, box/keypoint decoding, weighted NMS and the
//! ONNX wrapper.
//!
//! Ported from MediaPipe:
//! - `mediapipe/calculators/tflite/ssd_anchors_calculator.cc` (anchor layout)
//! - `mediapipe/calculators/tensor/tensors_to_detections_calculator.cc` (decode,
//!   `reverse_output_order`, sigmoid score with clipping)
//! - `mediapipe/calculators/util/non_max_suppression_calculator.cc` (`WEIGHTED`)
//! - `mediapipe/modules/palm_detection/palm_detection_cpu.pbtxt` (options)
//! - `mediapipe/calculators/tensor/image_to_tensor_calculator.cc` (letterbox)

use crate::config::{
    AnchorOptions, NUM_PALM_KEYPOINTS, PALM_BOX_VALUES, PALM_INPUT_SIZE, PALM_NMS_IOU,
    PALM_NUM_ANCHORS, PALM_SCORE_CLIP, PALM_SCORE_THRESHOLD,
};
use crate::frame::{Frame, Point, Rect, letterbox_rect, warp_to_tensor};
use crate::{Error, Result};
use ort::session::Session;
use ort::value::Tensor;

const MODEL_BYTES: &[u8] = include_bytes!("../models/hand_detector.onnx");

/// An SSD anchor, centre in `0..1` of the network input. With
/// `fixed_anchor_size` the size is always 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Anchor {
    /// Centre x.
    pub x_center: f64,
    /// Centre y.
    pub y_center: f64,
    /// Width.
    pub w: f64,
    /// Height.
    pub h: f64,
}

#[allow(clippy::cast_precision_loss, reason = "layer counts are tiny")]
fn calculate_scale(min_scale: f64, max_scale: f64, index: usize, num_strides: usize) -> f64 {
    if num_strides == 1 {
        return f64::midpoint(min_scale, max_scale);
    }
    min_scale + (max_scale - min_scale) * index as f64 / (num_strides - 1) as f64
}

/// Generate the SSD anchors (`SsdAnchorsCalculator::GenerateAnchors`).
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "feature-map sizes are tiny positive integers"
)]
pub fn generate_anchors(opts: &AnchorOptions) -> Vec<Anchor> {
    let n = opts.strides.len();
    let mut anchors = Vec::new();
    let mut layer = 0;
    while layer < n {
        let stride = opts.strides.get(layer).copied().unwrap_or(1);
        // All consecutive layers with the same stride share one feature map.
        let mut scales = Vec::new();
        let mut aspects = Vec::new();
        let mut last = layer;
        while last < n && opts.strides.get(last).copied() == Some(stride) {
            let scale = calculate_scale(opts.min_scale, opts.max_scale, last, n);
            for &ar in &opts.aspect_ratios {
                scales.push(scale);
                aspects.push(ar);
            }
            if opts.interpolated_scale_aspect_ratio > 0.0 {
                let next = if last == n - 1 {
                    1.0
                } else {
                    calculate_scale(opts.min_scale, opts.max_scale, last + 1, n)
                };
                scales.push((scale * next).sqrt());
                aspects.push(opts.interpolated_scale_aspect_ratio);
            }
            last += 1;
        }
        let fm_h = (f64::from(opts.input_height) / f64::from(stride)).ceil() as u32;
        let fm_w = (f64::from(opts.input_width) / f64::from(stride)).ceil() as u32;
        for y in 0..fm_h {
            for x in 0..fm_w {
                for (scale, ar) in scales.iter().zip(&aspects) {
                    let (w, h) = if opts.fixed_anchor_size {
                        (1.0, 1.0)
                    } else {
                        (scale * ar.sqrt(), scale / ar.sqrt())
                    };
                    anchors.push(Anchor {
                        x_center: (f64::from(x) + opts.anchor_offset_x) / f64::from(fm_w),
                        y_center: (f64::from(y) + opts.anchor_offset_y) / f64::from(fm_h),
                        w,
                        h,
                    });
                }
            }
        }
        layer = last;
    }
    anchors
}

/// The logistic function.
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// A palm detection. Coordinates are normalised `0..1` of the detector input
/// until [`remove_letterbox`] maps them to frame pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    /// Sigmoid score.
    pub score: f64,
    /// Left edge.
    pub xmin: f64,
    /// Top edge.
    pub ymin: f64,
    /// Right edge.
    pub xmax: f64,
    /// Bottom edge.
    pub ymax: f64,
    /// Wrist, index base, middle base, ring base, pinky base, thumb base, thumb tip (model order).
    pub keypoints: [Point; NUM_PALM_KEYPOINTS],
}

impl Detection {
    /// Intersection over union of the two boxes.
    #[must_use]
    pub fn iou(&self, other: &Self) -> f64 {
        let iw = (self.xmax.min(other.xmax) - self.xmin.max(other.xmin)).max(0.0);
        let ih = (self.ymax.min(other.ymax) - self.ymin.max(other.ymin)).max(0.0);
        let inter = iw * ih;
        let a = (self.xmax - self.xmin) * (self.ymax - self.ymin);
        let b = (other.xmax - other.xmin) * (other.ymax - other.ymin);
        let union = a + b - inter;
        if union <= 0.0 { 0.0 } else { inter / union }
    }
}

/// Decode raw detector outputs into detections above `score_threshold`.
///
/// `boxes` is `anchors.len() x 18` (x, y, w, h offsets then 7 keypoints as x, y
/// pairs; `reverse_output_order: true`), `scores` is `anchors.len()` raw logits.
/// `scale` is the detector input size (`x_scale`/`y_scale`/`w_scale`/`h_scale`).
#[must_use]
pub fn decode(
    boxes: &[f32],
    scores: &[f32],
    anchors: &[Anchor],
    scale: f64,
    score_threshold: f64,
) -> Vec<Detection> {
    let mut out = Vec::new();
    for ((raw, &logit), anchor) in boxes
        .as_chunks::<PALM_BOX_VALUES>()
        .0
        .iter()
        .zip(scores)
        .zip(anchors)
    {
        let score = sigmoid(f64::from(logit).clamp(-PALM_SCORE_CLIP, PALM_SCORE_CLIP));
        if score < score_threshold {
            continue;
        }
        let value = |i: usize| raw.get(i).copied().map_or(0.0, f64::from);
        let xc = value(0) / scale * anchor.w + anchor.x_center;
        let yc = value(1) / scale * anchor.h + anchor.y_center;
        let w = value(2) / scale * anchor.w;
        let h = value(3) / scale * anchor.h;
        let mut keypoints = [Point { x: 0.0, y: 0.0 }; NUM_PALM_KEYPOINTS];
        for (k, kp) in keypoints.iter_mut().enumerate() {
            kp.x = value(4 + 2 * k) / scale * anchor.w + anchor.x_center;
            kp.y = value(5 + 2 * k) / scale * anchor.h + anchor.y_center;
        }
        out.push(Detection {
            score,
            xmin: xc - w / 2.0,
            ymin: yc - h / 2.0,
            xmax: xc + w / 2.0,
            ymax: yc + h / 2.0,
            keypoints,
        });
    }
    out
}

/// Weighted non-max suppression (`NonMaxSuppressionCalculator`, `WEIGHTED`).
///
/// Detections are visited best first; each one absorbs every remaining
/// detection that overlaps it by more than `iou_threshold` and is replaced by
/// the score-weighted average of their boxes and keypoints. The output keeps the
/// best detection's score.
#[must_use]
pub fn weighted_nms(mut detections: Vec<Detection>, iou_threshold: f64) -> Vec<Detection> {
    detections.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut remaining = detections;
    let mut kept = Vec::new();
    while !remaining.is_empty() {
        let best = remaining.remove(0);
        let (mut group, rest): (Vec<_>, Vec<_>) = remaining
            .into_iter()
            .partition(|d| best.iou(d) > iou_threshold);
        remaining = rest;
        group.insert(0, best.clone());
        let total: f64 = group.iter().map(|d| d.score).sum();
        let mut merged = best;
        if group.len() > 1 && total > 0.0 {
            let avg = |f: &dyn Fn(&Detection) -> f64| {
                group.iter().map(|d| d.score * f(d)).sum::<f64>() / total
            };
            merged.xmin = avg(&|d| d.xmin);
            merged.ymin = avg(&|d| d.ymin);
            merged.xmax = avg(&|d| d.xmax);
            merged.ymax = avg(&|d| d.ymax);
            for (k, kp) in merged.keypoints.iter_mut().enumerate() {
                kp.x = avg(&|d| d.keypoints.get(k).map_or(0.0, |p| p.x));
                kp.y = avg(&|d| d.keypoints.get(k).map_or(0.0, |p| p.y));
            }
        }
        kept.push(merged);
    }
    kept
}

/// Map a detection from letterboxed detector coordinates (`0..1`) to frame
/// pixels (`DetectionLetterboxRemovalCalculator`).
#[must_use]
pub fn remove_letterbox(det: &Detection, letterbox: &Rect) -> Detection {
    let map = |p: Point| letterbox.norm_to_image(p.x, p.y);
    let top_left = map(Point {
        x: det.xmin,
        y: det.ymin,
    });
    let bottom_right = map(Point {
        x: det.xmax,
        y: det.ymax,
    });
    let mut keypoints = det.keypoints;
    for kp in &mut keypoints {
        *kp = map(*kp);
    }
    Detection {
        score: det.score,
        xmin: top_left.x,
        ymin: top_left.y,
        xmax: bottom_right.x,
        ymax: bottom_right.y,
        keypoints,
    }
}

/// The palm detector model plus its anchors.
pub struct PalmDetector {
    session: Session,
    anchors: Vec<Anchor>,
}

impl PalmDetector {
    /// Load the embedded `hand_detector.onnx`.
    ///
    /// # Errors
    /// Fails when ONNX Runtime cannot load the model or the anchor count does
    /// not match the model's output size.
    pub fn new() -> Result<Self> {
        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_memory(MODEL_BYTES)?;
        let anchors = generate_anchors(&AnchorOptions::default());
        if anchors.len() != PALM_NUM_ANCHORS {
            return Err(Error::ModelOutput {
                name: "anchors",
                detail: format!(
                    "generated {} anchors, expected {PALM_NUM_ANCHORS}",
                    anchors.len()
                ),
            });
        }
        Ok(Self { session, anchors })
    }

    /// Detect palms in `frame`. Results are in frame pixels, best first.
    ///
    /// # Errors
    /// Fails when inference fails or the output shapes are not the documented ones.
    pub fn detect(&mut self, frame: &Frame) -> Result<Vec<Detection>> {
        let letterbox = letterbox_rect(frame.width, frame.height, PALM_INPUT_SIZE, PALM_INPUT_SIZE);
        let data = warp_to_tensor(frame, &letterbox, PALM_INPUT_SIZE, PALM_INPUT_SIZE);
        let side = PALM_INPUT_SIZE as usize;
        let input = Tensor::from_array(([1usize, side, side, 3], data))?;
        let outputs = self.session.run(ort::inputs!["input_1" => input])?;
        let n = self.anchors.len();
        let (box_shape, boxes) = extract(&outputs, "Identity")?;
        let (score_shape, scores) = extract(&outputs, "Identity_1")?;
        if box_shape != [1, to_i64(n), to_i64(PALM_BOX_VALUES)] {
            return Err(Error::ModelOutput {
                name: "Identity",
                detail: format!("shape {box_shape:?}, expected [1, {n}, {PALM_BOX_VALUES}]"),
            });
        }
        if score_shape != [1, to_i64(n), 1] {
            return Err(Error::ModelOutput {
                name: "Identity_1",
                detail: format!("shape {score_shape:?}, expected [1, {n}, 1]"),
            });
        }
        let detections = decode(
            boxes,
            scores,
            &self.anchors,
            f64::from(PALM_INPUT_SIZE),
            PALM_SCORE_THRESHOLD,
        );
        Ok(weighted_nms(detections, PALM_NMS_IOU)
            .iter()
            .map(|d| remove_letterbox(d, &letterbox))
            .collect())
    }
}

fn to_i64(n: usize) -> i64 {
    i64::try_from(n).unwrap_or(i64::MAX)
}

/// Pull a named f32 output tensor out of a run result as (shape, data).
pub(crate) fn extract<'o>(
    outputs: &'o ort::session::SessionOutputs<'_>,
    name: &'static str,
) -> Result<(Vec<i64>, &'o [f32])> {
    let value = outputs.get(name).ok_or_else(|| Error::ModelOutput {
        name,
        detail: "missing from model outputs".to_owned(),
    })?;
    let (shape, data) = value.try_extract_tensor::<f32>()?;
    Ok((shape.iter().copied().collect(), data))
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test index arithmetic")]
mod tests {
    use super::*;

    fn anchors() -> Vec<Anchor> {
        generate_anchors(&AnchorOptions::default())
    }

    #[test]
    fn anchor_count_matches_the_model_output() {
        assert_eq!(anchors().len(), PALM_NUM_ANCHORS);
    }

    #[test]
    fn first_anchor_is_the_centre_of_the_first_stride_8_cell() {
        let a = anchors();
        let first = a.first().expect("anchors");
        // 192 / 8 = 24 cells; centre of cell (0, 0) is 0.5 / 24.
        assert!((first.x_center - 0.5 / 24.0).abs() < 1e-12);
        assert!((first.y_center - 0.5 / 24.0).abs() < 1e-12);
        assert!((first.w - 1.0).abs() < 1e-12 && (first.h - 1.0).abs() < 1e-12);
        // Two anchors per cell share a centre; the next cell is one cell to the right.
        let second = a.get(1).expect("anchor");
        assert!((second.x_center - first.x_center).abs() < 1e-12);
        let third = a.get(2).expect("anchor");
        assert!((third.x_center - 1.5 / 24.0).abs() < 1e-12);
    }

    #[test]
    fn stride_16_layers_start_after_the_24x24x2_grid() {
        let a = anchors();
        let first16 = a.get(24 * 24 * 2).expect("anchor");
        assert!((first16.x_center - 0.5 / 12.0).abs() < 1e-12);
        assert!((first16.y_center - 0.5 / 12.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_known_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!((sigmoid(2.0) - 0.880_797_077_977_882_4).abs() < 1e-12);
        assert!(sigmoid(-100.0) < 1e-40);
    }

    fn anchor_at(x: f64, y: f64) -> Anchor {
        Anchor {
            x_center: x,
            y_center: y,
            w: 1.0,
            h: 1.0,
        }
    }

    #[test]
    fn decode_applies_anchor_offsets_scores_and_threshold() {
        // Anchor 0 passes (logit 2), anchor 1 fails (logit -2).
        let anchors = [anchor_at(0.5, 0.5), anchor_at(0.25, 0.25)];
        let mut boxes = vec![0.0_f32; 2 * PALM_BOX_VALUES];
        // Anchor 0: dx=19.2px -> +0.1, dy=-19.2px, w=h=38.4px -> 0.2; keypoint 0 at +96px/+96px.
        for (i, v) in [19.2, -19.2, 38.4, 38.4].iter().enumerate() {
            boxes[i] = *v;
        }
        boxes[4] = 96.0;
        boxes[5] = 96.0;
        let scores = [2.0_f32, -2.0];
        let dets = decode(&boxes, &scores, &anchors, 192.0, 0.5);
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert!((d.score - sigmoid(2.0)).abs() < 1e-6);
        assert!((d.xmin - 0.5).abs() < 1e-5 && (d.xmax - 0.7).abs() < 1e-5);
        assert!((d.ymin - 0.3).abs() < 1e-5 && (d.ymax - 0.5).abs() < 1e-5);
        assert!((d.keypoints[0].x - 1.0).abs() < 1e-5 && (d.keypoints[0].y - 1.0).abs() < 1e-5);
    }

    #[test]
    fn decode_clips_extreme_logits() {
        let anchors = [anchor_at(0.5, 0.5)];
        let boxes = vec![0.0_f32; PALM_BOX_VALUES];
        let dets = decode(&boxes, &[1.0e9], &anchors, 192.0, 0.5);
        assert!((dets[0].score - 1.0).abs() < 1e-9);
    }

    fn det(score: f64, xmin: f64, ymin: f64, xmax: f64, ymax: f64) -> Detection {
        let kp = Point {
            x: f64::midpoint(xmin, xmax),
            y: f64::midpoint(ymin, ymax),
        };
        Detection {
            score,
            xmin,
            ymin,
            xmax,
            ymax,
            keypoints: [kp; NUM_PALM_KEYPOINTS],
        }
    }

    #[test]
    fn iou_of_known_boxes() {
        let a = det(1.0, 0.0, 0.0, 2.0, 2.0);
        let b = det(1.0, 1.0, 1.0, 3.0, 3.0);
        assert!((a.iou(&b) - 1.0 / 7.0).abs() < 1e-12);
        assert!((a.iou(&a) - 1.0).abs() < 1e-12);
        assert!(a.iou(&det(1.0, 5.0, 5.0, 6.0, 6.0)).abs() < 1e-12);
    }

    #[test]
    fn weighted_nms_merges_overlaps_by_score_weight() {
        let strong = det(0.9, 0.0, 0.0, 1.0, 1.0);
        let weak = det(0.3, 0.1, 0.0, 1.1, 1.0);
        let far = det(0.8, 5.0, 5.0, 6.0, 6.0);
        let out = weighted_nms(vec![weak, far, strong], 0.3);
        assert_eq!(out.len(), 2);
        // Best first; keeps the best score.
        assert!((out[0].score - 0.9).abs() < 1e-12);
        let expected_xmin = (0.9 * 0.0 + 0.3 * 0.1) / 1.2;
        assert!((out[0].xmin - expected_xmin).abs() < 1e-12);
        assert!((out[1].xmin - 5.0).abs() < 1e-12);
    }

    #[test]
    fn weighted_nms_keeps_non_overlapping_boxes_untouched() {
        let a = det(0.9, 0.0, 0.0, 1.0, 1.0);
        let b = det(0.7, 2.0, 0.0, 3.0, 1.0);
        let out = weighted_nms(vec![a.clone(), b.clone()], 0.3);
        assert_eq!(out, vec![a, b]);
    }

    #[test]
    fn letterbox_removal_maps_back_to_frame_pixels() {
        // 1280x720 into a square: padded to 1280x1280 centred on the frame.
        let lb = letterbox_rect(1280, 720, 192, 192);
        let d = det(0.9, 0.25, 0.5, 0.75, 0.75);
        let m = remove_letterbox(&d, &lb);
        assert!((m.xmin - 320.0).abs() < 1e-9 && (m.xmax - 960.0).abs() < 1e-9);
        assert!((m.ymin - 360.0).abs() < 1e-9 && (m.ymax - 680.0).abs() < 1e-9);
    }
}
