/// img.prep

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use fast_image_resize::{PixelType, Resizer, images::Image};
use image::{ColorType, DynamicImage, ImageFormat, ImageReader};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use walkdir::WalkDir;

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum, Debug)]
enum ResizeMode {
    /// suitable for ResNet / ViT
    Stretch,
    /// suitable for YOLO
    Letterbox,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {

    #[arg(short, long)]
    input: PathBuf,

    #[arg(short, long)]
    output: PathBuf,

    #[arg(short = 'W', long, default_value_t = 512)]
    width: u32,

    #[arg(short = 'H', long, default_value_t = 512)]
    height: u32,

    #[arg(short, long, value_enum, default_value_t = ResizeMode::Stretch)]
    mode: ResizeMode,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if !args.input.exists() || !args.input.is_dir() {
        anyhow::bail!(
            "Input folder does not exist: {:?}",
            args.input
        );
    }

    fs::create_dir_all(&args.output)?;

    println!("Scan files in {:?}...", args.input);
    let files: Vec<PathBuf> = WalkDir::new(&args.input)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.into_path())
        .filter(|p| is_supported_image(p))
        .collect();

    let total_files = files.len() as u64;
    if total_files == 0 {
        println!("No images found.");
        return Ok(());
    }

    let pb = ProgressBar::new(total_files);
    pb.set_style(
        ProgressStyle::with_template(
            "{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({per_sec}, ETA: {eta})",
        )?
        .progress_chars("#>-"),
    );

    let processed_count = AtomicUsize::new(0);
    let corrupted_count = AtomicUsize::new(0);

    // parallel processing
    files.par_iter().for_each_init(
        || Resizer::new(),
        |resizer, path| {
            match process_single_image(path, &args, resizer) {
                Ok(_) => {
                    processed_count.fetch_add(1, Ordering::Relaxed);
                }
                Err(_) => {
                    corrupted_count.fetch_add(1, Ordering::Relaxed);
                }
            }
            pb.inc(1);
        },
    );

    pb.finish_with_message("Processing is complete.");

    println!(
        "\nResult -> Successful: {}, Failed/skipped: {}",
        processed_count.load(Ordering::Relaxed),
        corrupted_count.load(Ordering::Relaxed)
    );

    Ok(())
}

fn is_supported_image(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_lowercase())
            .as_deref(),
        Some("jpg" | "jpeg" | "png" | "webp")
    )
}

fn process_single_image(src_path: &Path, args: &Args, resizer: &mut Resizer) -> Result<()> {
    let decoded = ImageReader::open(src_path)?
        .with_guessed_format()?
        .decode()
        .context("Corrupted file")?;

    // Uniform 3-channel RGBB (remove alpha channels)
    let rgb_image = DynamicImage::ImageRgb8(decoded.to_rgb8());

    let final_buffer = match args.mode {
        ResizeMode::Stretch => {
            let mut dst_image = Image::new(args.width, args.height, PixelType::U8x3);
            resizer.resize(&rgb_image, &mut dst_image, None)?;
            dst_image.into_vec()
        }
        ResizeMode::Letterbox => {
            resize_letterbox(&rgb_image, args.width, args.height, resizer)?
        }
    };

    let rel_path = src_path.strip_prefix(&args.input)?;
    let mut out_path = args.output.join(rel_path);
    out_path.set_extension("webp");

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent)?;
    }

    image::save_buffer_with_format(
        &out_path,
        &final_buffer,
        args.width,
        args.height,
        ColorType::Rgb8,
        ImageFormat::WebP,
    )?;

    Ok(())
}

fn resize_letterbox(
    src_image: &DynamicImage,
    target_w: u32,
    target_h: u32,
    resizer: &mut Resizer,
) -> Result<Vec<u8>> {
    let src_w = src_image.width() as f32;
    let src_h = src_image.height() as f32;

    let scale = (target_w as f32 / src_w).min(target_h as f32 / src_h);
    let new_w = ((src_w * scale).round() as u32).max(1).min(target_w);
    let new_h = ((src_h * scale).round() as u32).max(1).min(target_h);

    let mut scaled_img = Image::new(new_w, new_h, PixelType::U8x3);
    resizer.resize(src_image, &mut scaled_img, None)?;

    let mut canvas = vec![114u8; (target_w * target_h * 3) as usize];

    let offset_x = (target_w - new_w) / 2;
    let offset_y = (target_h - new_h) / 2;

    let src_buf = scaled_img.buffer();
    let src_stride = (new_w * 3) as usize;
    let dst_stride = (target_w * 3) as usize;

    for y in 0..new_h as usize {
        let src_start = y * src_stride;
        let src_end = src_start + src_stride;

        let dst_start = ((y + offset_y as usize) * dst_stride) + (offset_x as usize * 3);
        let dst_end = dst_start + src_stride;

        canvas[dst_start..dst_end].copy_from_slice(&src_buf[src_start..src_end]);
    }

    Ok(canvas)
}
