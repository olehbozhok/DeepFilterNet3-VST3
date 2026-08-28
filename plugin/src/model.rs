//! Worker-owned DeepFilterNet model state.
//!
//! `DfTract` is intentionally confined to this module and constructed by the
//! worker thread. The host callback only exchanges fixed-size audio messages.

use df::tract::{DfParams, DfTract, RuntimeParams};
use ndarray::Array2;
use std::path::{Path, PathBuf};

// Exactly one embedded model, or none. Two would mean the binary carries a
// model nobody asked for and the plugin's own report of its latency could
// describe the wrong one.
#[cfg(any(
    all(feature = "model-ll", feature = "model-standard"),
    all(feature = "model-ll", feature = "model-shortwave"),
    all(feature = "model-ll", feature = "model-custom"),
    all(feature = "model-standard", feature = "model-shortwave"),
    all(feature = "model-standard", feature = "model-custom"),
    all(feature = "model-shortwave", feature = "model-custom"),
))]
compile_error!(
    "enable at most one of model-ll, model-standard, model-shortwave, \n     model-custom - the default is model-ll, and any other needs \n     --no-default-features"
);

/// The archive compiled into this build, if any.
///
/// The path is resolved by build.rs - the bundled shortwave model by default,
/// or whatever `DEEPFILTER_EMBED_MODEL` names - and handed back through
/// `cargo:rustc-env`, because `include_bytes!` needs a literal and cannot make
/// that choice itself. build.rs also declares the resolved file as a build
/// input, so retraining the model rebuilds the plugin instead of leaving stale
/// weights behind a fresh path.
#[cfg(any(feature = "model-shortwave", feature = "model-custom"))]
const EMBEDDED_MODEL: &[u8] = include_bytes!(env!("DEEPFILTER_EMBED_MODEL"));

/// Whether this build carries a model at all.
pub(crate) const HAS_EMBEDDED_MODEL: bool = cfg!(any(
    feature = "model-ll",
    feature = "model-standard",
    feature = "model-shortwave",
    feature = "model-custom"
));

/// Fixed timing expected by the embedded official DeepFilterNet model.
pub(crate) const MODEL_SAMPLE_RATE: usize = 48_000;
pub(crate) const MODEL_HOP_SIZE: usize = 480;
const MIN_EFFECTIVE_ATTENUATION_DB: f32 = 0.01;

/// Immutable metadata derived from the constructed embedded model.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ModelInfo {
    pub(crate) sample_rate: usize,
    pub(crate) channels: usize,
    pub(crate) hop_size: usize,
    pub(crate) fft_size: usize,
    pub(crate) lookahead: usize,
    pub(crate) algorithmic_delay: usize,
}

impl ModelInfo {
    fn from_model(model: &DfTract) -> Result<Self, ModelError> {
        if model.ch != 1 {
            return Err(ModelError::new("DeepFilterNet model must have one channel"));
        }
        if model.sr != MODEL_SAMPLE_RATE {
            return Err(ModelError::new("DeepFilterNet model must run at 48 kHz"));
        }
        if model.hop_size == 0 || model.fft_size == 0 || model.fft_size < model.hop_size {
            return Err(ModelError::new("DeepFilterNet model has inconsistent frame sizes"));
        }

        let analysis_delay = model
            .fft_size
            .checked_sub(model.hop_size)
            .ok_or_else(|| ModelError::new("DeepFilterNet analysis delay underflowed"))?;
        let lookahead_delay = model
            .lookahead
            .checked_mul(model.hop_size)
            .ok_or_else(|| ModelError::new("DeepFilterNet lookahead delay overflowed"))?;
        let algorithmic_delay = analysis_delay
            .checked_add(lookahead_delay)
            .ok_or_else(|| ModelError::new("DeepFilterNet algorithmic delay overflowed"))?;

        Ok(Self {
            sample_rate: model.sr,
            channels: model.ch,
            hop_size: model.hop_size,
            fft_size: model.fft_size,
            lookahead: model.lookahead,
            algorithmic_delay,
        })
    }
}

/// Small owned error type kept on the worker side of the audio boundary.
#[derive(Debug)]
pub(super) struct ModelError(String);

impl ModelError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl std::fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// A non-`Send` model engine that exists only on the persistent worker thread.
pub(super) struct DfEngine {
    pristine: DfTract,
    active: DfTract,
    input: Array2<f32>,
    output: Array2<f32>,
    info: ModelInfo,
    last_applied_attenuation: Option<f32>,
}

/// Where a model comes from.
///
/// The embedded model is the default and the only one the shipped plugin needs.
/// The other two exist because a fine-tuned model is a file that changes: baking
/// it in means rebuilding the plugin for every training run, and comparing two
/// models by ear then means two builds. A path or a buffer costs nothing at
/// runtime and keeps that loop to seconds.
pub(crate) enum ModelSource<'a> {
    /// The model compiled in by the `model-ll` or `model-standard` feature.
    Embedded,
    /// A `*_onnx.tar.gz` on disk, as produced by DeepFilterNet's `export.py`.
    File(PathBuf),
    /// The same archive already in memory.
    Bytes(&'a [u8]),
}

/// Name of the environment variable that points at an external model.
///
/// An environment variable rather than a control in the editor: the editor is
/// specified to hold exactly two parameter sliders and nothing else, and a
/// model chooser is not a mix control. This is a developer's door, not a user's.
pub(crate) const MODEL_PATH_ENV: &str = "DEEPFILTER_MODEL";

#[cfg(any(feature = "model-shortwave", feature = "model-custom"))]
fn embedded_params() -> Result<DfParams, ModelError> {
    // Leaked for the same reason as ModelSource::Bytes, except that here the
    // slice is already 'static - it is in the binary - so nothing is leaked at
    // all. This is the cheapest of the three paths.
    DfParams::from_bytes(EMBEDDED_MODEL)
        .map_err(|error| ModelError::new(format!("the embedded model is unreadable: {error}")))
}

#[cfg(all(not(any(feature = "model-shortwave", feature = "model-custom")),
          any(feature = "model-ll", feature = "model-standard")))]
fn embedded_params() -> Result<DfParams, ModelError> {
    // DfParams::default() PANICS when DeepFilterNet was built without a model
    // feature, which is why this function is only compiled when one is present:
    // an audio plugin must fail as an error, never as a panic.
    Ok(DfParams::default())
}

#[cfg(not(any(feature = "model-ll", feature = "model-standard",
              feature = "model-shortwave", feature = "model-custom")))]
fn embedded_params() -> Result<DfParams, ModelError> {
    Err(ModelError::new(
        "this build carries no model. Set DEEPFILTER_MODEL to a *_onnx.tar.gz,          or rebuild with one of --features model-ll / model-standard / model-shortwave",
    ))
}

fn params_from(source: ModelSource<'_>) -> Result<DfParams, ModelError> {
    match source {
        ModelSource::Embedded => embedded_params(),
        ModelSource::File(path) => DfParams::new(path.clone()).map_err(|error| {
            ModelError::new(format!(
                "could not read the model at {}: {error}",
                path.display()
            ))
        }),
        ModelSource::Bytes(bytes) => {
            // `DfParams::from_bytes` wants `&'static [u8]`, because upstream
            // wrote it for `include_bytes!`. The archive has to outlive the
            // model that borrows from it, and the model lives as long as the
            // plugin instance, so the buffer is leaked deliberately - once, at
            // load, for about eight megabytes.
            //
            // That is a fair price for loading once. It would NOT be fair for a
            // plugin that swapped models repeatedly, so if that day comes this
            // is the line to revisit rather than the place to add a cache.
            let leaked: &'static [u8] = Box::leak(bytes.to_vec().into_boxed_slice());
            DfParams::from_bytes(leaked)
                .map_err(|error| ModelError::new(format!("could not read the model bytes: {error}")))
        }
    }
}

impl DfEngine {
    /// The embedded model, or whatever `DEEPFILTER_MODEL` points at.
    ///
    /// A missing or unreadable external model is an ERROR, not a silent
    /// fallback to the embedded one: someone who set the variable wants that
    /// model, and quietly running a different one would make every measurement
    /// afterwards a lie about which model produced it.
    pub(super) fn new() -> Result<Self, ModelError> {
        match std::env::var_os(MODEL_PATH_ENV) {
            Some(path) if !path.is_empty() => {
                Self::from_source(ModelSource::File(PathBuf::from(path)))
            }
            _ => Self::from_source(ModelSource::Embedded),
        }
    }

    /// Build from an archive already in memory.
    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, ModelError> {
        Self::from_source(ModelSource::Bytes(bytes))
    }

    /// Build from a `*_onnx.tar.gz` on disk.
    pub(crate) fn from_path(path: &Path) -> Result<Self, ModelError> {
        Self::from_source(ModelSource::File(path.to_path_buf()))
    }

    pub(crate) fn from_source(source: ModelSource<'_>) -> Result<Self, ModelError> {
        let params = params_from(source)?;
        let runtime = RuntimeParams::default_with_ch(1);
        let pristine = DfTract::new(params, &runtime)
            .map_err(|error| ModelError::new(format!("could not construct DeepFilterNet: {error}")))?;
        let info = ModelInfo::from_model(&pristine)?;
        let active = pristine.clone();
        let input = Array2::from_elem((1, info.hop_size), 0.0);
        let output = Array2::from_elem((1, info.hop_size), 0.0);

        Ok(Self {
            pristine,
            active,
            input,
            output,
            info,
            last_applied_attenuation: None,
        })
    }

    pub(super) fn info(&self) -> ModelInfo {
        self.info
    }

    /// Restore a pristine model and reusable frames before acknowledging reset.
    pub(super) fn reset(&mut self) {
        self.active = self.pristine.clone();
        self.input.fill(0.0);
        self.output.fill(0.0);
        self.last_applied_attenuation = None;
    }

    /// Process one exact mono model hop with the requested attenuation limit.
    pub(super) fn process_hop(
        &mut self,
        samples: &[f32],
        requested_attenuation: f32,
    ) -> Result<&[f32], ModelError> {
        if samples.len() != self.info.hop_size {
            return Err(ModelError::new("worker chunk does not match the model hop size"));
        }

        let effective_attenuation = effective_attenuation(requested_attenuation);
        if self.last_applied_attenuation != Some(effective_attenuation) {
            self.active.set_atten_lim(effective_attenuation);
            self.last_applied_attenuation = Some(effective_attenuation);
        }

        let input = self
            .input
            .as_slice_mut()
            .ok_or_else(|| ModelError::new("model input frame is not contiguous"))?;
        input.copy_from_slice(samples);
        self.active
            .process(self.input.view(), self.output.view_mut())
            .map_err(|error| ModelError::new(format!("DeepFilterNet hop processing failed: {error}")))?;

        self.output
            .as_slice()
            .ok_or_else(|| ModelError::new("model output frame is not contiguous"))
    }
}

pub(super) fn effective_attenuation(requested: f32) -> f32 {
    let requested = sanitized_attenuation(requested);
    requested.max(MIN_EFFECTIVE_ATTENUATION_DB)
}

/// Whether the user requested transparent enhancement while the model still advances.
pub(super) fn attenuation_is_effectively_zero(requested: f32) -> bool {
    sanitized_attenuation(requested) < MIN_EFFECTIVE_ATTENUATION_DB
}

fn sanitized_attenuation(requested: f32) -> f32 {
    if requested.is_finite() {
        requested.abs().min(100.0)
    } else {
        100.0
    }
}

// The module is compiled for EVERY build shape; the tests that need a
// particular embedded model gate themselves. Gating the whole module on
// model-ll meant the tests written for the other shapes silently did not
// exist - they reported "24 passed" while never being built.
#[cfg(test)]
mod tests {
    use super::*;

    fn model_fixture(sample_offset: usize) -> Vec<f32> {
        (0..MODEL_HOP_SIZE)
            .map(|index| {
                let phase = (sample_offset + index) as f32 / MODEL_SAMPLE_RATE as f32;
                0.15 * (phase * 440.0 * std::f32::consts::TAU).sin()
                    + 0.05 * (phase * 1_731.0 * std::f32::consts::TAU).sin()
            })
            .collect()
    }

    /// A build with no model must FAIL, and say what to do about it.
    ///
    /// The dangerous shape of this bug is not a crash: it is a plugin that
    /// silently falls back to some other model and passes audio, so every
    /// listening judgement afterwards is about a model nobody chose.
    #[cfg(not(any(feature = "model-ll", feature = "model-standard",
                  any(feature = "model-shortwave", feature = "model-custom"))))]
    #[test]
    fn a_build_with_no_model_refuses_clearly() {
        let error = DfEngine::from_source(ModelSource::Embedded)
            .err()
            .expect("a build with no model must not construct an engine");
        let text = error.to_string();
        assert!(text.contains("DEEPFILTER_MODEL"), "{text}");
        assert!(text.contains("model-shortwave"), "{text}");
    }

    /// An embedded model must be the one that was named, not a default.
    ///
    /// Skipped unless DEEPFILTER_TEST_MODEL names the SAME archive the build
    /// embedded: the test then asserts that loading it from disk and using the
    /// compiled-in copy describe the same model. A build that quietly fell back
    /// to DeepFilterNet's own model would differ in lookahead and fail here.
    #[cfg(any(feature = "model-shortwave", feature = "model-custom"))]
    #[test]
    fn the_embedded_model_is_the_one_that_was_named() {
        let _serial = crate::test_support::serialize_real_model();
        let embedded = DfEngine::from_source(ModelSource::Embedded)
            .expect("the embedded model must construct");
        let Some(path) = std::env::var_os("DEEPFILTER_TEST_MODEL") else {
            eprintln!("skipped the identity half: set DEEPFILTER_TEST_MODEL to                        the same archive that was embedded");
            return;
        };
        let from_disk = DfEngine::from_path(&PathBuf::from(path))
            .expect("the same archive must load from disk");
        assert_eq!(
            embedded.info(),
            from_disk.info(),
            "the compiled-in model does not match the archive it was built from"
        );
    }

    /// An external model has to load, report its own timing, and actually run.
    ///
    /// The point of the external path is a fine-tuned model, and a fine-tuned
    /// DeepFilterNet3 is architecturally the standard model - lookahead 2, not
    /// the low-latency 0 - so the timing it reports is NOT the embedded model's.
    /// A test that only checked "it constructs" would pass while the plugin
    /// reported the wrong latency to the host and aligned the dry path wrongly.
    ///
    /// Skipped, not failed, when no model file is given: it needs a real
    /// exported archive, and CI has none. Point `DEEPFILTER_TEST_MODEL` at a
    /// `*_onnx.tar.gz` to run it.
    #[test]
    fn an_external_model_loads_from_a_path_and_from_bytes() {
        // Works in any build shape: it never touches the embedded model.
        let Some(path) = std::env::var_os("DEEPFILTER_TEST_MODEL") else {
            eprintln!("skipped: set DEEPFILTER_TEST_MODEL to an exported *_onnx.tar.gz");
            return;
        };
        let path = PathBuf::from(path);
        let _serial = crate::test_support::serialize_real_model();

        let mut from_path = DfEngine::from_path(&path).expect("model must load from a path");
        let info = from_path.info();
        assert_eq!(info.sample_rate, MODEL_SAMPLE_RATE);
        assert_eq!(info.channels, 1);
        assert_eq!(info.hop_size, MODEL_HOP_SIZE);

        let bytes = std::fs::read(&path).expect("model file must be readable");
        let mut from_bytes = DfEngine::from_bytes(&bytes).expect("model must load from bytes");
        assert_eq!(
            from_bytes.info(),
            info,
            "the same archive must describe the same model whichever way it was read"
        );

        // And it must process, not merely construct. Silence in, silence out is
        // not evidence; a tone has to come through.
        let mut a_nonzero = false;
        let mut b_nonzero = false;
        for hop in 0..8 {
            let frame = model_fixture(hop * MODEL_HOP_SIZE);
            let a = from_path.process_hop(&frame, 20.0).expect("path model must process");
            a_nonzero |= a.iter().any(|sample| sample.abs() > 1e-6);
            let b = from_bytes.process_hop(&frame, 20.0).expect("bytes model must process");
            b_nonzero |= b.iter().any(|sample| sample.abs() > 1e-6);
        }
        assert!(a_nonzero && b_nonzero, "the external model produced only silence");
    }

    #[cfg(feature = "model-ll")]
    #[test]
    fn official_ll_metadata_and_one_channel_shape_are_live() {
        let _serial = crate::test_support::serialize_real_model();
        let mut engine = DfEngine::new().expect("official LL model must construct");
        assert_eq!(
            engine.info(),
            ModelInfo {
                sample_rate: 48_000,
                channels: 1,
                hop_size: 480,
                fft_size: 960,
                lookahead: 0,
                algorithmic_delay: 480,
            }
        );
        assert!(engine.process_hop(&[0.0; MODEL_HOP_SIZE - 1], 20.0).is_err());

        let mut observed_nonzero = false;
        for hop in 0..8 {
            let input = model_fixture(hop * MODEL_HOP_SIZE);
            let output = engine
                .process_hop(&input, 20.0)
                .expect("bounded one-channel inference must succeed");
            assert_eq!(output.len(), MODEL_HOP_SIZE);
            assert!(output.iter().all(|sample| sample.is_finite()));
            observed_nonzero |= output.iter().any(|sample| sample.abs() > 1.0e-8);
        }
        assert!(
            observed_nonzero,
            "bounded 20 dB model output must contain a non-silent sample"
        );
    }
}
