use std::path::Path;
use std::sync::{Mutex, OnceLock};

use image::imageops::FilterType;
use image::GrayImage;

/// Hard ceiling on decoded pixels — an anti-decompression-bomb guard, NOT
/// a size policy: only absurd inputs (an 8000x8000 claim that would
/// allocate gigabytes) are excluded. ~1.2 GB of RGB at the extreme.
const MAX_DECODE_PIXELS: u64 = 300_000_000;
/// Above this many pixels a decode is "heavy" (~100+ MB of RAM per image).
/// Heavy decodes are serialized across workers (one at a time) so four
/// 50 MP photos never decode simultaneously — but every image up to the
/// ceiling IS decoded and hashed. Nothing is skipped for being big.
const BIG_DECODE_PIXELS: u64 = 16_000_000;
/// Downscaled grayscale side for the DCT (32x32 is the pHash standard).
const DCT_SIZE: usize = 32;
/// 8x8 low-frequency block -> 64 output bits.
const HASH_BLOCK: usize = 8;
/// AC-coefficient spread below which an image counts as blank and its
/// perceptual hash as unreliable (see dct_phash).
const MIN_AC_SPREAD: f64 = 24.0;

fn big_decode_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Pixel dimensions read from the image header — cheap, no full decode.
/// Used by `--create` so it can record dimensions without hashing work.
pub fn dimensions_of(path: &Path) -> Option<(u32, u32)> {
    image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

/// Decode the image once and produce its 64-bit perceptual hash.
/// Returns `None` when the image doesn't meet the bar: undecodable
/// content or a blank hash-unreliable image. Big images are NOT excluded:
/// they are decoded with bounded memory pressure instead.
pub fn analyze(path: &Path) -> Option<i64> {
    // Cheap header read first: the dimensions decide the decode strategy
    // (and let absurd allocations be refused before they happen).
    let (width, height) = dims_of(path)?;
    let pixels = width as u64 * height as u64;
    if pixels > MAX_DECODE_PIXELS {
        return None;
    }

    // Heavy decodes go one-at-a-time across all workers; the guard lives
    // through decode + hash so the big buffers never overlap.
    let _heavy_guard = (pixels > BIG_DECODE_PIXELS)
        .then(|| big_decode_lock().lock().unwrap());

    let gray = decode_gray(path)?;
    dct_phash(&gray)
}

fn dims_of(path: &Path) -> Option<(u32, u32)> {
    image::ImageReader::open(path)
        .ok()?
        .with_guessed_format()
        .ok()?
        .into_dimensions()
        .ok()
}

fn decode_gray(path: &Path) -> Option<GrayImage> {
    let mut limits = image::Limits::default();
    // Backstop allocation cap ~ MAX_DECODE_PIXELS in RGB bytes.
    limits.max_alloc = Some(MAX_DECODE_PIXELS.saturating_mul(4));
    let mut reader = image::ImageReader::open(path).ok()?
        .with_guessed_format().ok()?;
    reader.limits(limits);
    let img = reader.decode().ok()?;
    Some(img.grayscale().into_luma8())
}

/// 64-bit DCT pHash: resize to 32x32 grayscale, separable DCT-II, keep
/// the top-left 8x8 low-frequency block, threshold each coefficient
/// against the median of its AC coefficients (DC excluded). Bit `i` of
/// the hash (MSB first) is coefficient `i` above the median.
///
/// Returns `None` for blank/near-blank images: when the AC spread is
/// negligible the median sits inside rounding noise and the hash is
/// meaningless (same rationale as PDQ's quality score — don't cluster
/// solid-color images).
fn dct_phash(gray: &GrayImage) -> Option<i64> {
    let n = DCT_SIZE;
    let resized = image::imageops::resize(gray, n as u32, n as u32, FilterType::Triangle);
    let pixels: Vec<f64> = resized.pixels().map(|p| p.0[0] as f64).collect();

    // Separable 2D DCT-II: transform rows, then columns.
    let mut tmp = vec![0.0f64; n * n];
    for y in 0..n {
        for u in 0..n {
            let mut s = 0.0;
            for x in 0..n {
                s += pixels[y * n + x] * dct_cos(x, u, n);
            }
            tmp[y * n + u] = s;
        }
    }
    let mut freq = vec![0.0f64; n * n];
    for x in 0..n {
        for v in 0..n {
            let mut s = 0.0;
            for y in 0..n {
                s += tmp[y * n + x] * dct_cos(y, v, n);
            }
            freq[v * n + x] = s;
        }
    }

    let mut coefs = [0.0f64; HASH_BLOCK * HASH_BLOCK];
    for v in 0..HASH_BLOCK {
        for u in 0..HASH_BLOCK {
            coefs[v * HASH_BLOCK + u] = freq[v * n + u];
        }
    }
    let mut ac: Vec<f64> = coefs[1..].to_vec();
    ac.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = ac[ac.len() / 2];
    if ac[ac.len() - 1] - ac[0] < MIN_AC_SPREAD {
        return None; // blank image: median thresholding is pure noise
    }

    let mut bits = 0u64;
    for (i, &c) in coefs.iter().enumerate() {
        if c > median {
            bits |= 1 << (63 - i);
        }
    }
    // Reinterpret the raw 64 bits as a signed int64: SQLite INTEGER is
    // two's-complement, so this is a lossless store of the hash. Unsigned
    // view = `value as u64` in Rust / `value & 0xFFFFFFFFFFFFFFFF` in SQL.
    Some(bits as i64)
}

/// Unnormalized DCT-II cosine for one input index and one frequency.
/// Scaling factors cancel out in the median thresholding.
fn dct_cos(x: usize, k: usize, n: usize) -> f64 {
    let scale = if k == 0 { std::f64::consts::FRAC_1_SQRT_2 } else { 1.0 };
    ((std::f64::consts::PI / n as f64) * (x as f64 + 0.5) * k as f64).cos() * scale
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    fn hamming(a: i64, b: i64) -> u32 {
        (a as u64 ^ b as u64).count_ones()
    }

    fn gradient(noise: i32, seed: u8) -> GrayImage {
        let mut img = GrayImage::new(64, 64);
        let mut s = seed as i32;
        for (i, pixel) in img.pixels_mut().enumerate() {
            let (x, y) = ((i % 64) as i32, (i / 64) as i32);
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let n = (s >> 16) % (noise + 1);
            *pixel = Luma([(255 - (x * 3 + y * 2) + n).clamp(0, 255) as u8]);
        }
        img
    }

    #[test]
    fn phash_is_deterministic() {
        let g = gradient(3, 1);
        assert_eq!(dct_phash(&g), dct_phash(&g));
    }

    #[test]
    fn near_identical_images_hash_closely() {
        let base = gradient(3, 1);
        // A slight blur mimics a re-encode/rescale: low frequencies (the
        // only ones pHash keeps) are preserved.
        let blurred = dct_phash(&image::imageops::blur(&base, 1.5f32))
            .expect("textured image must hash");
        let base = dct_phash(&base).expect("textured image must hash");
        let d = hamming(base, blurred);
        assert!(d <= 6, "blurred copy diverged: hamming {d}");
    }

    #[test]
    fn different_images_hash_far_apart() {
        let grad = dct_phash(&gradient(0, 1)).expect("textured image must hash");
        // Inverted brightness ramp: a genuinely different visual.
        let mut inv_img = GrayImage::new(64, 64);
        for (i, pixel) in inv_img.pixels_mut().enumerate() {
            let (x, y) = ((i % 64) as i32, (i / 64) as i32);
            *pixel = Luma([(x * 3 + y * 2).clamp(0, 255) as u8]);
        }
        let inv = dct_phash(&inv_img).expect("textured image must hash");
        assert!(hamming(grad, inv) >= 10, "distinct images too close");
    }

    #[test]
    fn blank_images_get_no_hash() {
        let flat = GrayImage::from_pixel(64, 64, Luma([200]));
        assert_eq!(dct_phash(&flat), None);
    }

    #[test]
    fn signed_store_preserves_bits() {
        // The unsigned->signed cast must be a pure bit reinterpretation.
        let bits: u64 = 0xFFFF_FFFF_FFFF_FFFF;
        assert_eq!(bits as i64, -1i64);
        assert_eq!((bits as i64 as u64), bits);
    }
}
