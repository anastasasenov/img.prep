## img_prep: Ultra-fast parallel image preprocessor and dataset converter written in Rust

[![License: MIT](https://shields.io)](https://opensource.org)
![AI Assisted](https://shields.io)

This experimental program is a high-performance CLI tool designed to prepare massive computer vision datasets (500,000+ images) before training deep learning models such as YOLO, ResNet, or ViT. By combining work-stealing parallelism with hardware SIMD instructions, it reduces dataset preprocessing time from hours to minutes while saving tens of gigabytes of disk space.


### Features

* Hardware-Accelerated Resizing: Uses fast_image_resize with SIMD instructions (AVX2, SSE4.1, NEON) for 10x-20x faster image scaling compared to standard Python libraries.
* True Multi-Core Parallelism: Leverages rayon with thread-local resizer reuse to saturate all available CPU cores without GIL bottlenecks.
* Flexible Resize Modes (stretch & letterbox):
  . stretch: Direct resizing to exact target dimensions (ideal for classification models like ResNet or ViT).
  . letterbox: Aspect-ratio-preserving scaling centered on a standard YOLO neutral gray.
* Channel Normalization: Automatically standardizes grayscale (Luma) and transparent (RGBA) images into uniform 3-channel RGB8 buffers.
* Automatic Corruption Filtering: Validates image headers and pixel buffers during decoding, safely skipping broken or truncated files.


### Installation & Build

#### Prerequisites
* [Rust toolchain](https://rustup.rs) (Edition 2024 / rustc 1.85+)

#### Building from Source
For maximum throughput, compile with native CPU optimizations enabled so the compiler can utilize the fastest SIMD instructions supported by your processor:

git clone https://github.com/xxx/img_prep
cd img_prep

RUSTFLAGS="-C target-cpu=native" cargo build --release

The compiled binary will be available at ./target/release/img_prep.

### Usage & CLI Options

#### Command-Line Arguments

    --input <PATH>    -i   Path to the source directory containing raw images
    --output <PATH>   -o   Path to the destination directory for processed images
    --width <PIXELS>  -W   Target image width in pixels, default 512
    --height <PIXELS> -H   Target image height in pixels, default 512
    --mode <MODE>     -m`  Resize strategy: stretch or letterbox, default stretch
    --help            -h   Print help information
    --version         -V   Print version information

#### Examples

Object Detection Dataset (YOLO - preserves aspect ratio with gray padding):

./target/release/img_prep -i ./raw_yolo -o ./ready_yolo -W 640 -H 640 --mode letterbox


Image Classification Dataset (ResNet / ViT - stretches to exact dimensions):

./target/release/img_prep -i ./raw_imagenet -o ./ready_imagenet -W 224 -H 224 --mode stretch


Quick Run with Default Settings (`512x512`, `stretch`):

./target/release/img_prep -i ./raw_dataset -o ./processed_dataset

### Folders

raw_dataset/                        processed_dataset/
    train/                              train/
        cats/                               cats/
            img_001.jpg  --->                   img_001.webp (640x640 RGB8)
        dogs/                               dogs/
            img_002.png  --->                   img_002.webp (640x640 RGB8)
    
### Tips

    Use an NVMe SSD: img_prep is fast enough that disk I/O often becomes the primary bottleneck. Placing both source and destination directories on an NVMe drive yields the highest throughput.

    Choose letterbox for Bounding Boxes: If your dataset includes object detection annotations (bounding boxes), use --mode letterbox so object proportions remain intact.

### Stack

    [clap](https://crates.io): Command-line argument parsing with derive macros.
    [rayon](https://crates.io): Data-parallelism library for distributing image workloads across all CPU threads.
    [fast_image_resize](https://crates.io): High-speed image resizer utilizing CPU vector extensions.
    (https://crates.io): Image decoding, format guessing, and WebP encoding.
    [indicatif](https://crates.io): Terminal progress bar and throughput metrics.
    [walkdir](https://crates.io): Fast recursive directory traversal.
    [anyhow](https://crates.io): Ergonomic error handling for isolating corrupted files.


