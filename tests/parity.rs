//! Fixture parity: Rust landmarks vs the official MediaPipe Python output
//! committed in `tests/fixtures/expected/` (made by `scripts/reference_landmarks.py`).
//!
//! The metric is the mean over a hand's 21 landmarks of the Euclidean distance
//! in normalised image coordinates (x / width, y / height).

use handwave::camera::load_image;
use handwave::config::{FilterConfig, PRESENCE_THRESHOLD};
use handwave::frame::Frame;
use handwave::landmarks::{Landmark, LandmarkModel};
use handwave::roi::rect_from_landmarks;
use handwave::tracker::Tracker;
use serde::Deserialize;
use std::error::Error;
use std::path::PathBuf;

type TestResult = Result<(), Box<dyn Error>>;

/// Per-landmark mean normalised error that must not be exceeded (from the spec).
const TOLERANCE: f64 = 0.02;

#[derive(Deserialize)]
struct Expected {
    width: u32,
    height: u32,
    hands: Vec<ExpectedHand>,
}

#[derive(Deserialize)]
struct ExpectedHand {
    landmarks: Vec<[f64; 3]>,
}

struct Fixture {
    name: String,
    frame: Frame,
    expected: Expected,
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn fixtures() -> Result<Vec<Fixture>, Box<dyn Error>> {
    let dir = root().join("tests/fixtures/hands");
    let mut names: Vec<String> = std::fs::read_dir(&dir)?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "png"))
        .filter_map(|p| Some(p.file_stem()?.to_str()?.to_owned()))
        .collect();
    names.sort();
    if names.is_empty() {
        return Err(format!("no fixtures in {}", dir.display()).into());
    }
    let mut out = Vec::new();
    for name in names {
        let frame = load_image(&dir.join(format!("{name}.png")))?;
        let json =
            std::fs::read_to_string(root().join(format!("tests/fixtures/expected/{name}.json")))?;
        let expected: Expected = serde_json::from_str(&json)?;
        if (expected.width, expected.height) != (frame.width, frame.height) {
            return Err(format!("{name}: reference size does not match the image").into());
        }
        out.push(Fixture {
            name,
            frame,
            expected,
        });
    }
    Ok(out)
}

fn normalised(frame: &Frame, landmarks: &[Landmark]) -> Vec<[f64; 2]> {
    landmarks
        .iter()
        .map(|l| [l.x / f64::from(frame.width), l.y / f64::from(frame.height)])
        .collect()
}

/// Mean normalised distance between two landmark sets of equal length.
fn mean_error(rust: &[[f64; 2]], expected: &[[f64; 3]]) -> f64 {
    let sum: f64 = rust
        .iter()
        .zip(expected)
        .map(|(r, e)| (r[0] - e[0]).hypot(r[1] - e[1]))
        .sum();
    let n = u32::try_from(rust.len()).map_or(f64::MAX, f64::from);
    sum / n
}

fn wrist_distance(rust: &[[f64; 2]], expected: &ExpectedHand) -> f64 {
    match (rust.first(), expected.landmarks.first()) {
        (Some(r), Some(e)) => (r[0] - e[0]).hypot(r[1] - e[1]),
        _ => f64::MAX,
    }
}

/// Match Rust hands to reference hands by wrist distance and return the
/// per-hand errors. Errors out when the hand counts differ.
fn match_errors(
    name: &str,
    rust: &[Vec<[f64; 2]>],
    expected: &Expected,
) -> Result<Vec<f64>, Box<dyn Error>> {
    if rust.len() != expected.hands.len() {
        return Err(format!(
            "{name}: found {} hand(s), reference has {}",
            rust.len(),
            expected.hands.len()
        )
        .into());
    }
    let mut free: Vec<&Vec<[f64; 2]>> = rust.iter().collect();
    let mut errors = Vec::new();
    for hand in &expected.hands {
        let (pos, _) = free
            .iter()
            .enumerate()
            .min_by(|a, b| wrist_distance(a.1, hand).total_cmp(&wrist_distance(b.1, hand)))
            .ok_or("no hand left to match")?;
        let best = free.remove(pos);
        errors.push(mean_error(best, &hand.landmarks));
    }
    Ok(errors)
}

#[test]
fn detection_path_matches_mediapipe_on_every_fixture() -> TestResult {
    let mut tracker = Tracker::new(FilterConfig::default())?;
    let mut worst: f64 = 0.0;
    for fixture in fixtures()? {
        tracker.reset();
        let result = tracker.process(&fixture.frame, 0.0)?;
        let rust: Vec<Vec<[f64; 2]>> = result
            .hands
            .iter()
            .map(|h| normalised(&fixture.frame, &h.raw))
            .collect();
        for (i, e) in match_errors(&fixture.name, &rust, &fixture.expected)?
            .iter()
            .enumerate()
        {
            println!(
                "parity[detect] {} hand {i}: mean normalised error {e:.5}",
                fixture.name
            );
            worst = worst.max(*e);
            assert!(
                *e <= TOLERANCE,
                "{} hand {i}: error {e} > {TOLERANCE}",
                fixture.name
            );
        }
    }
    println!("parity[detect] worst: {worst:.5}");
    Ok(())
}

/// The tracking path: run the landmark model on the ROI derived from the
/// previous landmarks (no palm detection) and compare with the reference again.
#[test]
fn tracking_roi_from_landmarks_stays_on_the_hand() -> TestResult {
    let mut tracker = Tracker::new(FilterConfig::default())?;
    let mut model = LandmarkModel::new()?;
    for fixture in fixtures()? {
        tracker.reset();
        let first = tracker.process(&fixture.frame, 0.0)?;
        let mut rust = Vec::new();
        for hand in &first.hands {
            let again = model.run(&fixture.frame, &rect_from_landmarks(&hand.raw))?;
            assert!(
                again.presence >= PRESENCE_THRESHOLD,
                "{}: hand lost on the tracking ROI",
                fixture.name
            );
            rust.push(normalised(&fixture.frame, &again.landmarks));
        }
        for (i, e) in match_errors(&fixture.name, &rust, &fixture.expected)?
            .iter()
            .enumerate()
        {
            println!(
                "parity[track]  {} hand {i}: mean normalised error {e:.5}",
                fixture.name
            );
            assert!(
                *e <= TOLERANCE,
                "{} hand {i}: tracking error {e} > {TOLERANCE}",
                fixture.name
            );
        }
    }
    Ok(())
}

#[test]
fn tracker_keeps_its_track_across_frames() -> TestResult {
    let fixture = fixtures()?
        .into_iter()
        .find(|f| f.name == "open_palm")
        .ok_or("open_palm fixture missing")?;
    let mut tracker = Tracker::new(FilterConfig::default())?;
    let first = tracker.process(&fixture.frame, 0.0)?;
    assert_eq!(first.hands.len(), 1);
    assert!(
        first.palm_ms.is_some(),
        "the first frame must run palm detection"
    );
    let second = tracker.process(&fixture.frame, 0.033)?;
    assert_eq!(
        second.hands.len(),
        1,
        "the same hand must not be tracked twice"
    );
    // The track continues from landmarks; it was not re-created from a detection.
    assert_eq!(
        first.hands.first().map(|h| h.id),
        second.hands.first().map(|h| h.id)
    );
    Ok(())
}
