//! SigLIP 2 model management and inference.
//!
//! Three files are pulled from the Hugging Face repo
//! `onnx-community/siglip2-base-patch32-256-ONNX` into the official HF
//! cache (respecting HF_HOME / HF_HUB_CACHE / HF_ENDPOINT / HF_TOKEN —
//! shared with `hf download`, transformers, and every other HF client):
//!
//! - `vision_model_q4f16.onnx` — image encoder for --generate-embeddings
//! - `text_model_q4f16.onnx`   — text encoder for --search
//! - `tokenizer.json`          — SentencePiece tokenizer for --search
//!
//! Two quirk notes carried from the model research
//! (docs/embedding-model-research.md): q4f16 instead of int8 because
//! dynamic-int8 exports use ConvInteger ops that ORT's CPU provider does
//! not implement; and GPU execution requires enabling ort
//! execution-provider cargo features and registering them on the session
//! (silent CPU fallback is ORT's own behavior).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use ort::environment::Environment;
use ort::session::{builder::GraphOptimizationLevel, Session, SessionOutputs};
use ort::value::Tensor;
use tokenizers::Tokenizer;

/// ort rc.14+: exactly one ORT environment per process; every session
/// builder takes it. Created lazily on the first session.
static ORT_ENV: OnceLock<Environment> = OnceLock::new();

fn ort_environment() -> Result<Environment, String> {
    if let Some(env) = ORT_ENV.get() {
        return Ok(env.clone());
    }
    let env = ort::init().build().map_err(|e| format!("ort init: {e}"))?;
    let _ = ORT_ENV.set(env.clone());
    Ok(env)
}

/// Repo paths inside onnx-community/siglip2-base-patch32-256-ONNX.
pub const VISION_FILE: &str = "onnx/vision_model_q4f16.onnx";
pub const TEXT_FILE: &str = "onnx/text_model_q4f16.onnx";
pub const TOKENIZER_FILE: &str = "tokenizer.json";

/// The text tower pads/truncates to 64 tokens.
const TEXT_MAX_TOKENS: usize = 64;

/// Repo the model files come from.
const REPO_ORG: &str = "onnx-community";
const REPO_NAME: &str = "siglip2-base-patch32-256-ONNX";

fn models_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
    home.join(".cache").join("imgindex").join("models")
}

/// Remove the pre-0.13 model cache (our own `~/.cache/imgindex/models`
/// directory). The model now lives in the official Hugging Face cache,
/// shared with `hf download`, transformers, and every other HF client.
fn cleanup_legacy_model_cache() {
    let dir = models_dir();
    let mut removed = false;
    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.filter_map(Result::ok) {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("siglip2-base-patch32-256-vision-")
                && std::fs::remove_file(entry.path()).is_ok()
            {
                removed = true;
            }
        }
        if removed {
            std::fs::remove_dir(&dir).ok(); // only succeeds when empty
        }
    }
    if removed {
        println!(
            "  removed legacy model cache at {} (model moved to the official Hugging Face cache)",
            dir.display()
        );
    }
}

/// Resolve one repo file through the Hugging Face cache: returns the
/// cached snapshot path, downloading on first use. Content-addressed,
/// with on-disk locking — concurrent fetches deduplicate.
pub fn ensure_file(repo_file: &str) -> Result<PathBuf, String> {
    let client = hf_hub::HFClientSync::new().map_err(|e| format!("hf client: {e}"))?;
    client
        .model(REPO_ORG, REPO_NAME)
        .download_file()
        .filename(repo_file)
        .send()
        .map_err(|e| format!("hf download ({repo_file}): {e}"))
}

/// Make sure all three model files are present in the HF cache. Runs the
/// legacy-cache cleanup; repeated calls are cache hits.
pub fn ensure_models() -> Result<(), String> {
    cleanup_legacy_model_cache();
    ensure_file(VISION_FILE)?;
    ensure_file(TEXT_FILE)?;
    ensure_file(TOKENIZER_FILE)?;
    Ok(())
}

fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| *x * *x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

fn new_session(model_path: &Path) -> Result<Session, String> {
    Session::builder(&ort_environment()?)
        .map_err(|e| format!("ort builder: {e}"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ort optimization: {e}"))?
        .with_intra_threads(4)
        .map_err(|e| format!("ort threads: {e}"))?
        .commit_from_file(model_path)
        .map_err(|e| format!("ort load ({}): {e}", model_path.display()))
}

/// Pull a 768-dim embedding out of a session output: use the vector
/// as-is when the output is [1, D]; mean-pool over the token axis when
/// it is [1, seq, D]; then L2-normalize.
fn finish_embedding(
    outputs: &SessionOutputs<'_>,
) -> Result<Vec<f32>, String> {
    // The trained projection head wins: SigLIP exports expose
    // pooler_output [1, D] next to the raw last_hidden_state [1, seq, D].
    // Mean-pooling hidden states would give untrained vectors.
    if let Some(v) = outputs.get("pooler_output") {
        let (shape, data) = v
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("ort output: {e}"))?;
        return Ok(finish(shape.to_vec(), data));
    }
    for (name, value) in outputs.iter() {
        if name.contains("embeds") {
            let (shape, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("ort output: {e}"))?;
            return Ok(finish(shape.to_vec(), data));
        }
    }
    let (shape, data) = outputs[0]
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("ort output: {e}"))?;
    Ok(finish(shape.to_vec(), data))
}

fn finish(shape: Vec<i64>, data: &[f32]) -> Vec<f32> {
    let mut embedding: Vec<f32> = if shape.len() == 3 {
        let seq = shape[1] as usize;
        let dims = shape[2] as usize;
        let mut pooled = vec![0f32; dims];
        for t in 0..seq {
            for d in 0..dims {
                pooled[d] += data[t * dims + d];
            }
        }
        for x in pooled.iter_mut() {
            *x /= seq as f32;
        }
        pooled
    } else {
        data.to_vec()
    };
    l2_normalize(&mut embedding);
    embedding
}

/// Vision encoder: image file -> 768-dim SigLIP 2 embedding.
pub struct Embedder {
    session: Session,
}

impl Embedder {
    pub fn new(model_path: &Path) -> Result<Self, String> {
        Ok(Self { session: new_session(model_path)? })
    }

    /// SigLIP 2 preprocessing: resize the shortest edge to 256 px
    /// (bilinear), center-crop to 256x256, scale to [0,1], normalize with
    /// mean/std 0.5 (the OpenCLIP-style normalization SigLIP models are
    /// trained with). Output is a CHW float tensor.
    pub fn embed(&mut self, path: &Path) -> Result<Vec<f32>, String> {
        const INPUT_SIZE: usize = 256;
        let img = image::ImageReader::open(path)
            .map_err(|e| format!("open: {e}"))?
            .with_guessed_format()
            .map_err(|e| format!("format: {e}"))?
            .decode()
            .map_err(|e| format!("decode: {e}"))?
            .to_rgb8();

        let (w, h) = (img.width(), img.height());
        let resized = if w != h || w != INPUT_SIZE as u32 {
            // shortest edge to INPUT_SIZE, keep aspect, then center-crop
            let scale = if w < h {
                INPUT_SIZE as f32 / w as f32
            } else {
                INPUT_SIZE as f32 / h as f32
            };
            let (nw, nh) =
                ((w as f32 * scale).round() as u32, (h as f32 * scale).round() as u32);
            let resized = image::imageops::resize(
                &img,
                nw.max(1),
                nh.max(1),
                image::imageops::FilterType::Triangle,
            );
            let (rw, rh) = (resized.width(), resized.height());
            let cx = rw.saturating_sub(INPUT_SIZE as u32) / 2;
            let cy = rh.saturating_sub(INPUT_SIZE as u32) / 2;
            image::imageops::crop_imm(
                &resized,
                cx,
                cy,
                INPUT_SIZE as u32,
                INPUT_SIZE as u32,
            )
            .to_image()
        } else {
            img
        };

        let n = INPUT_SIZE * INPUT_SIZE;
        let mut tensor = vec![0f32; 3 * n];
        for (i, p) in resized.pixels().enumerate() {
            let [r, g, b] = p.0;
            tensor[i] = (r as f32 / 255.0 - 0.5) / 0.5;
            tensor[n + i] = (g as f32 / 255.0 - 0.5) / 0.5;
            tensor[2 * n + i] = (b as f32 / 255.0 - 0.5) / 0.5;
        }

        let input = Tensor::from_array((
            [1i64, 3, INPUT_SIZE as i64, INPUT_SIZE as i64],
            tensor,
        ))
        .map_err(|e| format!("ort tensor: {e}"))?;

        let outputs = self
            .session
            .run(ort::inputs!["pixel_values" => input])
            .map_err(|e| format!("ort run: {e}"))?;
        finish_embedding(&outputs)
    }
}

/// Text encoder: query string -> 768-dim SigLIP 2 embedding.
pub struct TextEmbedder {
    session: Session,
    tokenizer: Tokenizer,
    takes_attention_mask: bool,
}

impl TextEmbedder {
    pub fn new(model_path: &Path, tokenizer_path: &Path) -> Result<Self, String> {
        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("tokenizer load: {e}"))?;
        // The exported graph was traced at the model's fixed context
        // length; pad/truncate every query to exactly TEXT_MAX_TOKENS.
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: TEXT_MAX_TOKENS,
                stride: 0,
                strategy: tokenizers::TruncationStrategy::LongestFirst,
                direction: tokenizers::TruncationDirection::Right,
            }))
            .map_err(|e| format!("tokenizer truncation: {e}"))?;
        let pad_id = tokenizer.token_to_id("<pad>").unwrap_or(0);
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(TEXT_MAX_TOKENS),
            direction: tokenizers::PaddingDirection::Right,
            pad_to_multiple_of: None,
            pad_id,
            pad_type_id: 0,
            pad_token: "<pad>".into(),
        }));

        let session = new_session(model_path)?;
        let takes_attention_mask =
            session.inputs().iter().any(|i| i.name() == "attention_mask");
        Ok(Self { session, tokenizer, takes_attention_mask })
    }

    pub fn embed_text(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenize: {e}"))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| v as i64)
            .collect();

        let ids_tensor = Tensor::from_array(([1i64, TEXT_MAX_TOKENS as i64], ids))
            .map_err(|e| format!("ort tensor: {e}"))?;
        let outputs = if self.takes_attention_mask {
            let mask_tensor = Tensor::from_array(([1i64, TEXT_MAX_TOKENS as i64], mask))
                .map_err(|e| format!("ort tensor: {e}"))?;
            self.session
                .run(ort::inputs![
                    "input_ids" => ids_tensor,
                    "attention_mask" => mask_tensor
                ])
                .map_err(|e| format!("ort run: {e}"))?
        } else {
            self.session
                .run(ort::inputs!["input_ids" => ids_tensor])
                .map_err(|e| format!("ort run: {e}"))?
        };
        finish_embedding(&outputs)
    }
}
