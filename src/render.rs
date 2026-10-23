//! Drawing: the macroquad live view with its HUD, and the CPU annotator used
//! by `--image`.

use crate::config::{WINDOW_HEIGHT, WINDOW_WIDTH};
use crate::frame::Frame;
use crate::landmarks::Landmark;
use crate::tracker::TrackedHand;
use crate::worker::{self, Controls, JitterStatus, LatestSlot, Snapshot};
use crate::{Error, Result};
use macroquad::prelude as mq;
use std::sync::Arc;
use std::sync::atomic::Ordering;

/// The standard 21-point MediaPipe hand topology (wrist 0; thumb 1-4, index
/// 5-8, middle 9-12, ring 13-16, pinky 17-20).
pub const HAND_CONNECTIONS: [(usize, usize); 21] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
];

/// Line/dot colour of the nth hand.
const HAND_COLORS: [[u8; 3]; 2] = [[40, 230, 255], [255, 80, 200]];

fn hand_color(index: usize) -> [u8; 3] {
    HAND_COLORS
        .get(index % HAND_COLORS.len())
        .copied()
        .unwrap_or([255, 255, 255])
}

// ---------------------------------------------------------------- CPU annotator

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "pixel coordinates; every stamp is bounds-checked"
)]
fn stamp_disc(img: &mut image::RgbImage, cx: f64, cy: f64, radius: f64, rgb: [u8; 3]) {
    let r = radius.ceil() as i64 + 1;
    let (w, h) = (i64::from(img.width()), i64::from(img.height()));
    let (x0, y0) = (cx.round() as i64, cy.round() as i64);
    for y in (y0 - r)..=(y0 + r) {
        for x in (x0 - r)..=(x0 + r) {
            if x < 0 || y < 0 || x >= w || y >= h {
                continue;
            }
            let (dx, dy) = (x as f64 - cx, y as f64 - cy);
            if dx * dx + dy * dy <= radius * radius {
                img.put_pixel(x as u32, y as u32, image::Rgb(rgb));
            }
        }
    }
}

fn draw_segment(img: &mut image::RgbImage, a: &Landmark, b: &Landmark, radius: f64, rgb: [u8; 3]) {
    let len = (b.x - a.x).hypot(b.y - a.y);
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "small positive step count"
    )]
    let steps = (len / (radius * 0.5).max(0.5)).ceil().max(1.0) as u32;
    for i in 0..=steps {
        let t = f64::from(i) / f64::from(steps);
        stamp_disc(
            img,
            a.x + (b.x - a.x) * t,
            a.y + (b.y - a.y) * t,
            radius,
            rgb,
        );
    }
}

/// A copy of `frame` with the hands' topology and 21 points drawn on it.
///
/// # Errors
/// Fails when the frame buffer is inconsistent.
pub fn annotate(frame: &Frame, hands: &[TrackedHand]) -> Result<image::RgbImage> {
    let mut img = image::RgbImage::from_raw(frame.width, frame.height, frame.data.clone())
        .ok_or_else(|| Error::InvalidFrame("buffer does not match its size".to_owned()))?;
    let unit = f64::from(frame.width.max(frame.height)) / 400.0;
    let line_radius = (unit * 0.5).max(1.5);
    let dot_radius = (unit * 1.2).max(3.0);
    for (i, hand) in hands.iter().enumerate() {
        let color = hand_color(i);
        for &(a, b) in &HAND_CONNECTIONS {
            if let (Some(pa), Some(pb)) = (hand.raw.get(a), hand.raw.get(b)) {
                draw_segment(&mut img, pa, pb, line_radius, color);
            }
        }
        for lm in &hand.raw {
            stamp_disc(&mut img, lm.x, lm.y, dot_radius, [255, 255, 255]);
            stamp_disc(&mut img, lm.x, lm.y, dot_radius * 0.6, color);
        }
    }
    Ok(img)
}

// ---------------------------------------------------------------- live view

fn mq_color(rgb: [u8; 3], alpha: f32) -> mq::Color {
    mq::Color::new(
        f32::from(rgb[0]) / 255.0,
        f32::from(rgb[1]) / 255.0,
        f32::from(rgb[2]) / 255.0,
        alpha,
    )
}

/// Where the video lands in the window: scale and top-left offset.
#[derive(Clone, Copy)]
struct Placement {
    scale: f32,
    x: f32,
    y: f32,
}

impl Placement {
    #[allow(clippy::cast_precision_loss, reason = "frame sizes are small")]
    fn fit(frame: &Frame) -> Self {
        let (sw, sh) = (mq::screen_width(), mq::screen_height());
        let scale = (sw / frame.width as f32).min(sh / frame.height as f32);
        Self {
            scale,
            x: (sw - frame.width as f32 * scale) / 2.0,
            y: (sh - frame.height as f32 * scale) / 2.0,
        }
    }

    #[allow(
        clippy::cast_possible_truncation,
        reason = "screen coordinates fit f32"
    )]
    fn point(self, lm: &Landmark) -> (f32, f32) {
        (
            self.x + lm.x as f32 * self.scale,
            self.y + lm.y as f32 * self.scale,
        )
    }
}

fn draw_hand(landmarks: &[Landmark; 21], color: [u8; 3], placement: Placement) {
    // Soft glow first (wide, faint), then the crisp core.
    for &(width, alpha) in &[(14.0_f32, 0.07_f32), (8.0, 0.16), (3.0, 0.95)] {
        for &(a, b) in &HAND_CONNECTIONS {
            if let (Some(pa), Some(pb)) = (landmarks.get(a), landmarks.get(b)) {
                let (x0, y0) = placement.point(pa);
                let (x1, y1) = placement.point(pb);
                mq::draw_line(x0, y0, x1, y1, width, mq_color(color, alpha));
            }
        }
    }
    for lm in landmarks {
        let (x, y) = placement.point(lm);
        mq::draw_circle(x, y, 13.0, mq_color(color, 0.10));
        mq::draw_circle(x, y, 8.0, mq_color(color, 0.25));
        mq::draw_circle(x, y, 4.5, mq::WHITE);
        mq::draw_circle(x, y, 3.0, mq_color(color, 1.0));
    }
}

fn opt_ms(v: Option<f64>) -> String {
    v.map_or_else(|| "  -  ".to_owned(), |v| format!("{v:5.1}"))
}

fn hud_lines(snapshot: &Snapshot, filter_on: bool) -> Vec<String> {
    let mut lines = vec![format!(
        "cam {:4.1} fps | inf {:4.1} fps | palm {} ms | landmark {} ms | hands {} | {} | filter {}",
        snapshot.camera_fps,
        snapshot.inference_fps,
        opt_ms(snapshot.palm_ms),
        opt_ms(snapshot.landmark_ms),
        snapshot.hands.len(),
        snapshot.mode.label(),
        if filter_on { "on" } else { "off" },
    )];
    lines.push(match snapshot.jitter {
        JitterStatus::Idle => "J: jitter test (hold your hand still)".to_owned(),
        JitterStatus::Recording(left) => format!("jitter test: hold still... {left:.1}s"),
        JitterStatus::NoHand => "jitter test: no hand tracked".to_owned(),
        JitterStatus::Done(r) => format!(
            "jitter RMS raw {:.2} px | filtered {:.2} px ({} frames)",
            r.raw_rms, r.filtered_rms, r.samples
        ),
    });
    lines
}

/// A monospace system font if one can be loaded.
fn load_mono_font() -> Option<mq::Font> {
    [
        "/System/Library/Fonts/Monaco.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    .iter()
    .find_map(|path| {
        let bytes = std::fs::read(path).ok()?;
        mq::load_ttf_font_from_bytes(&bytes).ok()
    })
}

fn draw_text(font: Option<&mq::Font>, text: &str, x: f32, y: f32, color: mq::Color) {
    mq::draw_text_ex(
        text,
        x,
        y,
        mq::TextParams {
            font,
            font_size: 18,
            color,
            ..Default::default()
        },
    );
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "frame sizes fit u16 for textures"
)]
fn texture_from(frame: &Frame) -> mq::Texture2D {
    let mut rgba = Vec::with_capacity(frame.data.len() / 3 * 4);
    for px in frame.data.as_chunks::<3>().0 {
        rgba.extend_from_slice(px);
        rgba.push(255);
    }
    let texture = mq::Texture2D::from_rgba8(frame.width as u16, frame.height as u16, &rgba);
    texture.set_filter(mq::FilterMode::Linear);
    texture
}

async fn live_loop(controls: Arc<Controls>, slot: Arc<LatestSlot<Snapshot>>) {
    let font = load_mono_font();
    let mut latest: Option<Snapshot> = None;
    let mut texture: Option<mq::Texture2D> = None;
    loop {
        if mq::is_key_pressed(mq::KeyCode::Escape) {
            break;
        }
        if mq::is_key_pressed(mq::KeyCode::F) {
            controls.filter_on.fetch_xor(true, Ordering::Relaxed);
        }
        if mq::is_key_pressed(mq::KeyCode::J) {
            controls.jitter_request.store(true, Ordering::Relaxed);
        }
        if let Some(snapshot) = slot.take() {
            texture = Some(texture_from(&snapshot.frame));
            latest = Some(snapshot);
        }
        let filter_on = controls.filter_on.load(Ordering::Relaxed);

        mq::clear_background(mq::Color::new(0.04, 0.045, 0.06, 1.0));
        if let (Some(snapshot), Some(texture)) = (&latest, &texture) {
            let placement = Placement::fit(&snapshot.frame);
            #[allow(clippy::cast_precision_loss, reason = "frame sizes are small")]
            let size = mq::vec2(
                snapshot.frame.width as f32 * placement.scale,
                snapshot.frame.height as f32 * placement.scale,
            );
            mq::draw_texture_ex(
                texture,
                placement.x,
                placement.y,
                mq::WHITE,
                mq::DrawTextureParams {
                    dest_size: Some(size),
                    ..Default::default()
                },
            );
            for (i, hand) in snapshot.hands.iter().enumerate() {
                let points = if filter_on { &hand.smoothed } else { &hand.raw };
                draw_hand(points, hand_color(i), placement);
            }
            for (row, line) in hud_lines(snapshot, filter_on).iter().enumerate() {
                #[allow(clippy::cast_precision_loss, reason = "row index is tiny")]
                let y = 26.0 + row as f32 * 22.0;
                draw_text(
                    font.as_ref(),
                    line,
                    14.0,
                    y,
                    mq::Color::new(0.85, 1.0, 0.9, 1.0),
                );
            }
        } else if let Some(message) = controls.error_message() {
            draw_text(
                font.as_ref(),
                &format!("error: {message}"),
                14.0,
                26.0,
                mq::RED,
            );
        } else {
            draw_text(
                font.as_ref(),
                "starting camera... (allow camera access if macOS asks)",
                14.0,
                26.0,
                mq::GRAY,
            );
        }
        if latest.is_some()
            && let Some(message) = controls.error_message()
        {
            draw_text(
                font.as_ref(),
                &format!("error: {message}"),
                14.0,
                70.0,
                mq::RED,
            );
        }
        mq::next_frame().await;
    }
    controls.quit.store(true, Ordering::Relaxed);
}

/// Open the window and run until Esc or the window is closed.
///
/// # Errors
/// Fails when the worker thread cannot be started.
pub fn run_live() -> Result<()> {
    let controls = Arc::new(Controls::new());
    let slot = Arc::new(LatestSlot::default());
    let worker = worker::spawn(Arc::clone(&controls), Arc::clone(&slot))?;
    let conf = mq::Conf {
        window_title: "handwave".to_owned(),
        window_width: WINDOW_WIDTH,
        window_height: WINDOW_HEIGHT,
        ..Default::default()
    };
    macroquad::Window::from_config(conf, live_loop(Arc::clone(&controls), slot));
    controls.quit.store(true, Ordering::Relaxed);
    if worker.join().is_err() {
        return Err(Error::Camera("worker thread panicked".to_owned()));
    }
    if let Some(message) = controls.error_message() {
        return Err(Error::Camera(message));
    }
    Ok(())
}
