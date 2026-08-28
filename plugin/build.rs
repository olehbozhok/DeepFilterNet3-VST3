//! Resolve which model archive gets compiled in, and make it a build input.
//!
//! `include_bytes!` needs a string literal, so "the environment variable if it
//! is set, otherwise the one in this repository" cannot be decided in the
//! source. It is decided here and handed back as `DEEPFILTER_EMBED_MODEL`,
//! which the source then reads with `env!`.
//!
//! Both the variable and the resolved file are declared as build inputs.
//! Without that, retraining the model - or pointing the variable somewhere new -
//! leaves a binary that describes the new model and contains the old weights,
//! which is a difference nobody finds by listening.

use std::path::PathBuf;

const BUNDLED: &str = "models/dfn3-shortwave-v1_onnx.tar.gz";

fn main() {
    println!("cargo:rerun-if-env-changed=DEEPFILTER_EMBED_MODEL");

    if std::env::var_os("CARGO_FEATURE_MODEL_SHORTWAVE").is_none() {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let path = match std::env::var("DEEPFILTER_EMBED_MODEL") {
        Ok(p) if !p.is_empty() => PathBuf::from(p),
        _ => manifest.join(BUNDLED),
    };

    if !path.is_file() {
        // A missing archive has to stop the build. Falling back to another
        // model would ship weights nobody chose.
        panic!(
            "model-shortwave: no model archive at {}. Either restore {BUNDLED} \
             or set DEEPFILTER_EMBED_MODEL to a *_onnx.tar.gz exported by \
             DeepFilterNet's export.py",
            path.display()
        );
    }

    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo:rustc-env=DEEPFILTER_EMBED_MODEL={}", path.display());
}
