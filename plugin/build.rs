//! Resolve which model archive gets compiled in, and make it a build input.
//!
//! `include_bytes!` needs a string literal, so the choice cannot be made in the
//! source. It is made here and handed back as `DEEPFILTER_EMBED_MODEL`, which
//! the source reads with `env!`.
//!
//! TWO FEATURES, NOT ONE WITH A HIDDEN SWITCH:
//!
//!   model-shortwave  embeds the archive in this repository. Always that one.
//!                    Reproducible from a clone with nothing in the environment.
//!   model-custom     embeds whatever DEEPFILTER_EMBED_MODEL names, and fails
//!                    the build if it names nothing.
//!
//! They were one feature briefly, with the variable silently overriding the
//! bundled model. That meant two people could run the same build command and
//! get different weights, with nothing in the command to say so - the same
//! class of silent substitution this project has been bitten by repeatedly.
//!
//! Both the variable and the resolved file are declared as build inputs, so
//! retraining a model rebuilds the plugin instead of leaving stale weights
//! behind a fresh path.

use std::path::PathBuf;

const BUNDLED: &str = "models/dfn3-shortwave-v1_onnx.tar.gz";
const ENV: &str = "DEEPFILTER_EMBED_MODEL";

fn main() {
    println!("cargo:rerun-if-env-changed={ENV}");

    let bundled = std::env::var_os("CARGO_FEATURE_MODEL_SHORTWAVE").is_some();
    let custom = std::env::var_os("CARGO_FEATURE_MODEL_CUSTOM").is_some();
    if !bundled && !custom {
        return;
    }

    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let given = std::env::var(ENV).ok().filter(|p| !p.is_empty());

    let path = if custom {
        match given {
            Some(p) => PathBuf::from(p),
            None => panic!(
                "model-custom needs {ENV} to name a *_onnx.tar.gz exported by \
                 DeepFilterNet's export.py. To embed the model that ships with \
                 this repository instead, build with --features model-shortwave"
            ),
        }
    } else {
        // Deliberately ignores the variable: this feature means one specific
        // model. Say so rather than quietly doing what was asked elsewhere.
        if given.is_some() {
            println!(
                "cargo:warning={ENV} is set but --features model-shortwave \
                 embeds {BUNDLED} and ignores it. Use --features model-custom \
                 to embed the archive the variable names."
            );
        }
        manifest.join(BUNDLED)
    };

    if !path.is_file() {
        // A missing archive stops the build. Falling back to another model
        // would ship weights nobody chose.
        panic!("no model archive at {}", path.display());
    }

    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo:rustc-env={ENV}={}", path.display());
}
