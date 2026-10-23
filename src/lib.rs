//! handwave: native webcam hand tracking.
//!
//! The pipeline is a port of MediaPipe's hand landmarker (palm detector, hand
//! landmark model, ROI tracking) running the official models as ONNX through
//! ONNX Runtime. Everything runs locally; there are no network calls.

pub mod camera;
pub mod config;
pub mod filter;
pub mod frame;
pub mod landmarks;
pub mod palm;
pub mod render;
pub mod roi;
pub mod tracker;
pub mod worker;

/// The single error type of the crate.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// ONNX Runtime failed to load a model or run inference.
    #[error("inference error: {0}")]
    Inference(String),
    /// A model returned a tensor that does not have the documented shape.
    #[error("unexpected model output `{name}`: {detail}")]
    ModelOutput {
        /// Output tensor name.
        name: &'static str,
        /// What was wrong with it.
        detail: String,
    },
    /// The camera could not be opened or read.
    #[error("camera error: {0}")]
    Camera(String),
    /// Camera access was not granted by the user.
    #[error(
        "camera access was not granted; enable it in System Settings > Privacy & Security > Camera"
    )]
    CameraPermission,
    /// An image buffer did not match its declared size.
    #[error("invalid frame: {0}")]
    InvalidFrame(String),
    /// Reading or decoding an image file failed.
    #[error("image error: {0}")]
    Image(#[from] image::ImageError),
    /// A file operation failed.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Result alias using the crate [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

impl<R> From<ort::Error<R>> for Error {
    fn from(e: ort::Error<R>) -> Self {
        Self::Inference(e.to_string())
    }
}
