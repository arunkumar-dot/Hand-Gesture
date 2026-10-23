//! handwave CLI: live window by default, `--image` for one photo, `--bench`.

use clap::Parser;
use handwave::camera::load_image;
use handwave::config::{BENCH_PASSES, FilterConfig};
use handwave::render::{annotate, run_live};
use handwave::tracker::Tracker;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

/// Native webcam hand tracking.
#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    /// Run the pipeline on one image and save an annotated PNG (needs --out).
    #[arg(long, value_name = "INPUT", requires = "out", conflicts_with = "bench")]
    image: Option<PathBuf>,
    /// Where to write the annotated PNG.
    #[arg(long, value_name = "OUTPUT", requires = "image")]
    out: Option<PathBuf>,
    /// Time palm detection and landmarks over the fixtures.
    #[arg(long)]
    bench: bool,
}

fn annotate_one(input: &Path, output: &Path) -> handwave::Result<()> {
    // No mirroring here: the output must line up with the file as stored.
    let frame = load_image(input)?;
    let mut tracker = Tracker::new(FilterConfig::default())?;
    let result = tracker.process(&frame, 0.0)?;
    println!("{}: {} hand(s)", input.display(), result.hands.len());
    for hand in &result.hands {
        println!(
            "  hand {} presence {:.3} wrist ({:.1}, {:.1})",
            hand.id,
            hand.presence,
            hand.raw.first().map_or(0.0, |l| l.x),
            hand.raw.first().map_or(0.0, |l| l.y),
        );
    }
    annotate(&frame, &result.hands)?.save(output)?;
    println!("wrote {}", output.display());
    Ok(())
}

fn fixtures() -> handwave::Result<Vec<PathBuf>> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/hands");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension()
                .is_some_and(|x| x == "png" || x == "jpg" || x == "jpeg")
        })
        .collect();
    paths.sort();
    Ok(paths)
}

fn percentile95(sorted: &[f64]) -> f64 {
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "index of a short sample list"
    )]
    let idx = ((sorted.len() as f64) * 0.95).ceil() as usize;
    sorted
        .get(idx.saturating_sub(1))
        .or_else(|| sorted.last())
        .copied()
        .unwrap_or(0.0)
}

fn report(name: &str, mut samples: Vec<f64>) {
    samples.sort_by(f64::total_cmp);
    #[allow(clippy::cast_precision_loss, reason = "sample count is small")]
    let mean = samples.iter().sum::<f64>() / samples.len().max(1) as f64;
    println!(
        "{name:<10} n={:<5} mean {mean:7.2} ms  p95 {:7.2} ms",
        samples.len(),
        percentile95(&samples)
    );
}

fn bench() -> handwave::Result<()> {
    let frames = fixtures()?
        .iter()
        .map(|p| load_image(p))
        .collect::<handwave::Result<Vec<_>>>()?;
    println!("{BENCH_PASSES} passes over {} fixture(s)", frames.len());
    let mut tracker = Tracker::new(FilterConfig::default())?;
    let mut palm = Vec::new();
    let mut landmarks = Vec::new();
    let start = Instant::now();
    for _ in 0..BENCH_PASSES {
        for frame in &frames {
            // A fresh tracker per image makes every run detect, then run landmarks.
            tracker.reset();
            let result = tracker.process(frame, 0.0)?;
            palm.extend(result.palm_ms);
            landmarks.extend(result.landmark_ms);
        }
    }
    report("palm", palm);
    report("landmarks", landmarks);
    println!("total {:.1} s", start.elapsed().as_secs_f64());
    Ok(())
}

fn run() -> handwave::Result<()> {
    let cli = Cli::parse();
    match (cli.image, cli.out) {
        (Some(input), Some(output)) => annotate_one(&input, &output),
        _ if cli.bench => bench(),
        _ => run_live(),
    }
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
