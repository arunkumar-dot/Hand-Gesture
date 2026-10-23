//! Every tunable constant in one place.
//!
//! Model sizes come from `models/MODELS.md` (read from the converted ONNX files).
//! MediaPipe constants cite the graph they were read from; the ROI and anchor
//! values were checked against the serialized subgraph configs inside the
//! pinned `mediapipe` Python wheel (see `models/MODELS.md`).

/// Palm detector input side in pixels (`hand_detector.onnx`: `input_1` is 1x192x192x3).
pub const PALM_INPUT_SIZE: u32 = 192;
/// Landmark model input side in pixels (`hand_landmarks_detector.onnx`: `input_1` is 1x224x224x3).
pub const LANDMARK_INPUT_SIZE: u32 = 224;
/// Number of landmarks per hand.
pub const NUM_LANDMARKS: usize = 21;
/// Number of keypoints the palm detector regresses per detection.
pub const NUM_PALM_KEYPOINTS: usize = 7;

/// Parameters of MediaPipe's `SsdAnchorsCalculator` for the palm detector
/// (`palm_detection_cpu.pbtxt`).
#[derive(Debug, Clone)]
pub struct AnchorOptions {
    /// Network input width in pixels.
    pub input_width: u32,
    /// Network input height in pixels.
    pub input_height: u32,
    /// Smallest anchor scale.
    pub min_scale: f64,
    /// Largest anchor scale.
    pub max_scale: f64,
    /// Anchor centre offset within a cell, x.
    pub anchor_offset_x: f64,
    /// Anchor centre offset within a cell, y.
    pub anchor_offset_y: f64,
    /// Stride of each feature-map layer.
    pub strides: Vec<u32>,
    /// Aspect ratios of the anchors in every layer.
    pub aspect_ratios: Vec<f64>,
    /// Aspect ratio of the extra interpolated-scale anchor (0 disables it).
    pub interpolated_scale_aspect_ratio: f64,
    /// Every anchor is 1x1 when set (only centres matter).
    pub fixed_anchor_size: bool,
}

impl Default for AnchorOptions {
    fn default() -> Self {
        Self {
            input_width: PALM_INPUT_SIZE,
            input_height: PALM_INPUT_SIZE,
            min_scale: 0.148_437_5,
            max_scale: 0.75,
            anchor_offset_x: 0.5,
            anchor_offset_y: 0.5,
            strides: vec![8, 16, 16, 16],
            aspect_ratios: vec![1.0],
            interpolated_scale_aspect_ratio: 1.0,
            fixed_anchor_size: true,
        }
    }
}

/// Expected anchor count; also the second dimension of the detector outputs.
pub const PALM_NUM_ANCHORS: usize = 2016;
/// Values per anchor in the detector's box tensor (4 box + 7 keypoints x 2).
pub const PALM_BOX_VALUES: usize = 18;
/// Palm score threshold (after sigmoid).
pub const PALM_SCORE_THRESHOLD: f64 = 0.5;
/// Raw scores are clipped to +-this before the sigmoid.
pub const PALM_SCORE_CLIP: f64 = 100.0;
/// IoU above which weighted NMS merges detections.
pub const PALM_NMS_IOU: f64 = 0.3;

/// Detection -> ROI scale (`PalmDetectionDetectionToRoi`: `RectTransformation`).
pub const PALM_ROI_SCALE: f64 = 2.6;
/// Detection -> ROI vertical shift, in ROI heights.
pub const PALM_ROI_SHIFT_Y: f64 = -0.5;
/// Landmarks -> ROI scale (`HandLandmarkLandmarksToRoi`: `RectTransformation`).
pub const LANDMARK_ROI_SCALE: f64 = 2.0;
/// Landmarks -> ROI vertical shift, in ROI heights.
pub const LANDMARK_ROI_SHIFT_Y: f64 = -0.1;

/// Presence below this means the hand is lost.
pub const PRESENCE_THRESHOLD: f32 = 0.5;
/// Maximum number of tracked hands.
pub const MAX_HANDS: usize = 2;
/// A new detection whose ROI overlaps an existing track by more than this IoU is ignored.
pub const DUPLICATE_IOU: f64 = 0.5;

/// One Euro filter parameters (Casiez et al. 2012).
#[derive(Debug, Clone, Copy)]
pub struct FilterConfig {
    /// Cutoff frequency at zero speed, in Hz.
    pub min_cutoff: f64,
    /// Speed coefficient: how fast the cutoff opens up with speed.
    pub beta: f64,
    /// Cutoff frequency of the speed low-pass, in Hz.
    pub d_cutoff: f64,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            min_cutoff: 1.0,
            beta: 0.007,
            d_cutoff: 1.0,
        }
    }
}

/// Live window width.
pub const WINDOW_WIDTH: i32 = 1280;
/// Live window height.
pub const WINDOW_HEIGHT: i32 = 720;
/// Requested camera width.
pub const CAMERA_WIDTH: u32 = 1280;
/// Requested camera height.
pub const CAMERA_HEIGHT: u32 = 720;
/// Index of the camera to open.
pub const CAMERA_INDEX: u32 = 0;
/// Seconds the user holds still during the jitter test.
pub const JITTER_TEST_SECONDS: f64 = 3.0;
/// Landmarks averaged to get the palm centre: wrist and the four finger bases.
pub const PALM_CENTER_LANDMARKS: [usize; 5] = [0, 5, 9, 13, 17];
/// Passes over the fixtures in `--bench`.
pub const BENCH_PASSES: usize = 100;
