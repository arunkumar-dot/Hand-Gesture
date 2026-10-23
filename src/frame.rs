//! Pure image and geometry helpers: mirroring, letterboxing, affine warps and
//! coordinate transforms. No I/O, no inference.
//!
//! All geometry is in pixels of the source frame, y down, `f64`. Rotations are
//! in radians, positive clockwise on screen, matching MediaPipe's
//! `NormalizedRect.rotation`.
//!
//! The crop convention follows MediaPipe's `ImageToTensorCalculator`
//! (`image_to_tensor_converter_opencv.cc`): crop pixel `(x, y)` samples the
//! source at the point whose normalised position inside the rotated rect is
//! `(x / out_w, y / out_h)`, zero padding outside the frame.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    reason = "pixel coordinates and sizes are far below the range where these casts lose information"
)]

use crate::{Error, Result};

/// A packed RGB8 image.
#[derive(Debug, Clone)]
pub struct Frame {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// `width * height * 3` bytes, row-major RGB.
    pub data: Vec<u8>,
}

impl Frame {
    /// Wrap an RGB8 buffer.
    ///
    /// # Errors
    /// Returns [`Error::InvalidFrame`] when `data` is not `width * height * 3` bytes.
    pub fn new(width: u32, height: u32, data: Vec<u8>) -> Result<Self> {
        let expected = width as usize * height as usize * 3;
        if data.len() != expected {
            return Err(Error::InvalidFrame(format!(
                "{width}x{height} RGB needs {expected} bytes, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// The frame flipped left-right.
    #[must_use]
    pub fn mirrored(&self) -> Self {
        let row_bytes = self.width as usize * 3;
        let mut data = Vec::with_capacity(self.data.len());
        if row_bytes > 0 {
            for row in self.data.chunks_exact(row_bytes) {
                for px in row.as_chunks::<3>().0.iter().rev() {
                    data.extend_from_slice(px);
                }
            }
        }
        Self {
            width: self.width,
            height: self.height,
            data,
        }
    }

    /// RGB at integer coordinates, `[0, 0, 0]` outside the frame.
    fn pixel(&self, x: i64, y: i64) -> [f32; 3] {
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return [0.0; 3];
        }
        let offset = (y as usize * self.width as usize + x as usize) * 3;
        match self.data.get(offset..offset + 3) {
            Some(&[red, green, blue]) => [f32::from(red), f32::from(green), f32::from(blue)],
            _ => [0.0; 3],
        }
    }
}

/// A point in pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    /// Horizontal position.
    pub x: f64,
    /// Vertical position.
    pub y: f64,
}

/// A rotated rectangle in pixels (MediaPipe's `NormalizedRect`, un-normalised).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// Centre x.
    pub cx: f64,
    /// Centre y.
    pub cy: f64,
    /// Width before rotation.
    pub w: f64,
    /// Height before rotation.
    pub h: f64,
    /// Clockwise rotation in radians.
    pub rotation: f64,
}

impl Rect {
    /// Map a position normalised inside the rect (`0..1`, rect-local, before
    /// rotation) to frame pixels. This is `LandmarkProjectionCalculator`.
    #[must_use]
    pub fn norm_to_image(&self, u: f64, v: f64) -> Point {
        let (sin, cos) = self.rotation.sin_cos();
        let dx = (u - 0.5) * self.w;
        let dy = (v - 0.5) * self.h;
        Point {
            x: self.cx + cos * dx - sin * dy,
            y: self.cy + sin * dx + cos * dy,
        }
    }

    /// The affine map from crop pixels (`out_w` x `out_h`) to frame pixels.
    #[must_use]
    pub fn crop_to_image(&self, out_w: u32, out_h: u32) -> Affine {
        let (sin, cos) = self.rotation.sin_cos();
        let ow = f64::from(out_w);
        let oh = f64::from(out_h);
        Affine {
            a: cos * self.w / ow,
            b: -sin * self.h / oh,
            tx: self.cx - cos * 0.5 * self.w + sin * 0.5 * self.h,
            c: sin * self.w / ow,
            d: cos * self.h / oh,
            ty: self.cy - sin * 0.5 * self.w - cos * 0.5 * self.h,
        }
    }

    /// The axis-aligned bounds `(min_x, min_y, max_x, max_y)` of the rotated rect.
    #[must_use]
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        let (sin, cos) = self.rotation.sin_cos();
        let half_w = f64::midpoint(cos.abs() * self.w, sin.abs() * self.h);
        let half_h = f64::midpoint(sin.abs() * self.w, cos.abs() * self.h);
        (
            self.cx - half_w,
            self.cy - half_h,
            self.cx + half_w,
            self.cy + half_h,
        )
    }
}

/// A 2D affine transform `(x, y) -> (a*x + b*y + tx, c*x + d*y + ty)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Affine {
    /// Row 0, column 0.
    pub a: f64,
    /// Row 0, column 1.
    pub b: f64,
    /// Row 0 translation.
    pub tx: f64,
    /// Row 1, column 0.
    pub c: f64,
    /// Row 1, column 1.
    pub d: f64,
    /// Row 1 translation.
    pub ty: f64,
}

impl Affine {
    /// Apply the transform to a point.
    #[must_use]
    pub fn apply(&self, x: f64, y: f64) -> Point {
        Point {
            x: self.a * x + self.b * y + self.tx,
            y: self.c * x + self.d * y + self.ty,
        }
    }

    /// The inverse transform, or `None` when it is singular.
    #[must_use]
    pub fn invert(&self) -> Option<Self> {
        let det = self.a * self.d - self.b * self.c;
        if det.abs() < 1e-12 {
            return None;
        }
        let a = self.d / det;
        let b = -self.b / det;
        let c = -self.c / det;
        let d = self.a / det;
        Some(Self {
            a,
            b,
            c,
            d,
            tx: -(a * self.tx + b * self.ty),
            ty: -(c * self.tx + d * self.ty),
        })
    }
}

/// The rect that letterboxes a `width` x `height` frame into an `out_w` x
/// `out_h` tensor: full frame, padded to the output aspect ratio.
#[must_use]
pub fn letterbox_rect(width: u32, height: u32, out_w: u32, out_h: u32) -> Rect {
    let w = f64::from(width);
    let h = f64::from(height);
    let target_aspect = f64::from(out_w) / f64::from(out_h);
    let (rw, rh) = if w / h > target_aspect {
        (w, w / target_aspect)
    } else {
        (h * target_aspect, h)
    };
    Rect {
        cx: w / 2.0,
        cy: h / 2.0,
        w: rw,
        h: rh,
        rotation: 0.0,
    }
}

/// Warp the `rect` of `frame` into an `out_w` x `out_h` NHWC float tensor with
/// values in `0..1` (both models normalise as `x / 255`, see `models/MODELS.md`).
/// Bilinear sampling, zero padding outside the frame.
#[must_use]
pub fn warp_to_tensor(frame: &Frame, rect: &Rect, out_w: u32, out_h: u32) -> Vec<f32> {
    let map = rect.crop_to_image(out_w, out_h);
    let mut out = Vec::with_capacity(out_w as usize * out_h as usize * 3);
    for y in 0..out_h {
        for x in 0..out_w {
            let p = map.apply(f64::from(x), f64::from(y));
            let fx = p.x.floor();
            let fy = p.y.floor();
            let wx = (p.x - fx) as f32;
            let wy = (p.y - fy) as f32;
            let x0 = fx as i64;
            let y0 = fy as i64;
            let p00 = frame.pixel(x0, y0);
            let p10 = frame.pixel(x0 + 1, y0);
            let p01 = frame.pixel(x0, y0 + 1);
            let p11 = frame.pixel(x0 + 1, y0 + 1);
            for ch in 0..3 {
                let top = lerp(channel(&p00, ch), channel(&p10, ch), wx);
                let bottom = lerp(channel(&p01, ch), channel(&p11, ch), wx);
                out.push(lerp(top, bottom, wy) / 255.0);
            }
        }
    }
    out
}

fn channel(px: &[f32; 3], ch: usize) -> f32 {
    px.get(ch).copied().unwrap_or(0.0)
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss, reason = "test index arithmetic")]
mod tests {
    use super::*;

    fn gradient(width: u32, height: u32) -> Frame {
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                data.extend_from_slice(&[(x * 255 / width) as u8, (y * 255 / height) as u8, 7]);
            }
        }
        Frame::new(width, height, data).expect("valid frame")
    }

    #[test]
    fn frame_rejects_wrong_length() {
        assert!(Frame::new(2, 2, vec![0; 11]).is_err());
    }

    #[test]
    fn mirror_flips_columns_and_is_an_involution() {
        let f = Frame::new(2, 1, vec![1, 2, 3, 4, 5, 6]).expect("valid frame");
        let m = f.mirrored();
        assert_eq!(m.data, vec![4, 5, 6, 1, 2, 3]);
        assert_eq!(m.mirrored().data, f.data);
    }

    #[test]
    fn affine_projects_into_crop_and_back() {
        let rect = Rect {
            cx: 640.0,
            cy: 360.0,
            w: 300.0,
            h: 300.0,
            rotation: 0.7,
        };
        let crop_to_image = rect.crop_to_image(224, 224);
        let image_to_crop = crop_to_image.invert().expect("invertible");
        for &(x, y) in &[(0.0, 0.0), (111.5, 40.25), (224.0, 224.0), (3.0, 200.0)] {
            let p = crop_to_image.apply(x, y);
            let back = image_to_crop.apply(p.x, p.y);
            assert!((back.x - x).abs() < 1e-4 && (back.y - y).abs() < 1e-4);
        }
        let q = Point { x: 700.0, y: 400.0 };
        let c = image_to_crop.apply(q.x, q.y);
        let r = crop_to_image.apply(c.x, c.y);
        assert!((r.x - q.x).abs() < 1e-4 && (r.y - q.y).abs() < 1e-4);
    }

    #[test]
    fn norm_to_image_agrees_with_the_crop_affine() {
        let rect = Rect {
            cx: 100.0,
            cy: 80.0,
            w: 60.0,
            h: 40.0,
            rotation: -1.1,
        };
        let a = rect.crop_to_image(10, 20).apply(3.0, 5.0);
        let b = rect.norm_to_image(0.3, 0.25);
        assert!((a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9);
    }

    #[test]
    fn letterbox_pads_the_short_side() {
        let r = letterbox_rect(1280, 720, 192, 192);
        assert!((r.w - 1280.0).abs() < 1e-9 && (r.h - 1280.0).abs() < 1e-9);
        let r = letterbox_rect(720, 1280, 192, 192);
        assert!((r.w - 1280.0).abs() < 1e-9 && (r.h - 1280.0).abs() < 1e-9);
        // A pixel of the letterboxed tensor outside the frame maps outside it.
        assert!(r.norm_to_image(0.0, 0.5).x < 0.0);
    }

    #[test]
    fn identity_warp_reproduces_pixels() {
        let f = gradient(16, 16);
        let rect = Rect {
            cx: 8.0,
            cy: 8.0,
            w: 16.0,
            h: 16.0,
            rotation: 0.0,
        };
        let t = warp_to_tensor(&f, &rect, 16, 16);
        assert_eq!(t.len(), 16 * 16 * 3);
        let i = (5 * 16 + 9) * 3;
        let want = f.data.get(i..i + 3).expect("pixel");
        let got = t.get(i..i + 3).expect("tensor pixel");
        for (w, g) in want.iter().zip(got) {
            assert!((f32::from(*w) / 255.0 - g).abs() < 1e-5);
        }
    }

    #[test]
    fn warp_pads_with_zeros_outside_the_frame() {
        let f = Frame::new(4, 4, vec![255; 48]).expect("valid frame");
        let rect = Rect {
            cx: 100.0,
            cy: 100.0,
            w: 4.0,
            h: 4.0,
            rotation: 0.0,
        };
        assert!(warp_to_tensor(&f, &rect, 4, 4).iter().all(|v| *v == 0.0));
    }

    #[test]
    fn bounds_of_a_quarter_turn_swap_width_and_height() {
        let rect = Rect {
            cx: 0.0,
            cy: 0.0,
            w: 4.0,
            h: 2.0,
            rotation: std::f64::consts::FRAC_PI_2,
        };
        let (x0, y0, x1, y1) = rect.bounds();
        assert!((x1 - x0 - 2.0).abs() < 1e-9 && (y1 - y0 - 4.0).abs() < 1e-9);
    }
}
