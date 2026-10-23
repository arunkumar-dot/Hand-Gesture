//! Webcam capture (nokhwa, AVFoundation backend) producing RGB frames.

use crate::config::{CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH};
use crate::frame::Frame;
use crate::{Error, Result};
use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType, Resolution};
use std::time::{Duration, Instant};

/// How long to wait for the user to answer the macOS camera permission prompt.
const PERMISSION_TIMEOUT: Duration = Duration::from_secs(60);

fn camera_error(e: &nokhwa::NokhwaError) -> Error {
    Error::Camera(e.to_string())
}

/// An open, streaming camera.
pub struct Camera {
    inner: nokhwa::Camera,
}

impl Camera {
    /// Ask macOS for camera access (first run only) and start streaming,
    /// preferring 1280x720.
    ///
    /// The returned camera is not `Send`; open it on the thread that reads it.
    ///
    /// # Errors
    /// [`Error::CameraPermission`] when access is denied or the prompt is not
    /// answered in time, [`Error::Camera`] for any other failure.
    pub fn open() -> Result<Self> {
        // macOS: nokhwa must be initialised (which triggers the permission
        // prompt) before AVFoundation is touched.
        nokhwa::nokhwa_initialize(|_granted| {});
        let deadline = Instant::now() + PERMISSION_TIMEOUT;
        while !nokhwa::nokhwa_check() {
            if Instant::now() >= deadline {
                return Err(Error::CameraPermission);
            }
            std::thread::sleep(Duration::from_millis(100));
        }

        let wanted = Resolution::new(CAMERA_WIDTH, CAMERA_HEIGHT);
        let open = |kind| {
            let format = RequestedFormat::new::<RgbFormat>(kind);
            nokhwa::Camera::new(CameraIndex::Index(CAMERA_INDEX), format)
        };
        let mut inner = open(RequestedFormatType::HighestResolution(wanted))
            .or_else(|_| open(RequestedFormatType::AbsoluteHighestFrameRate))
            .map_err(|e| camera_error(&e))?;
        inner.open_stream().map_err(|e| camera_error(&e))?;
        Ok(Self { inner })
    }

    /// Block until the next frame and decode it to RGB.
    ///
    /// # Errors
    /// [`Error::Camera`] when capture or decoding fails.
    pub fn next_frame(&mut self) -> Result<Frame> {
        let buffer = self.inner.frame().map_err(|e| camera_error(&e))?;
        let image = buffer
            .decode_image::<RgbFormat>()
            .map_err(|e| camera_error(&e))?;
        let (width, height) = (image.width(), image.height());
        Frame::new(width, height, image.into_raw())
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // Nothing useful can be done if stopping fails while dropping.
        let _ = self.inner.stop_stream();
    }
}

/// Load an image file as an RGB frame (the `--image` mode and the fixtures).
///
/// # Errors
/// Fails when the file cannot be read or decoded.
pub fn load_image(path: &std::path::Path) -> Result<Frame> {
    let image = image::open(path)?.to_rgb8();
    let (width, height) = (image.width(), image.height());
    Frame::new(width, height, image.into_raw())
}
