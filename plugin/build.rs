//! Build script: make an embedded model's path a build input.
//!
//! `include_bytes!(env!("DEEPFILTER_EMBED_MODEL"))` is evaluated at compile
//! time, so cargo has to be told that the variable and the file it names are
//! inputs. Without this, changing which model is embedded - or retraining the
//! one already named - leaves a stale binary that reports the new path and
//! contains the old weights, which is the kind of difference nobody finds by
//! listening.

fn main() {
    println!("cargo:rerun-if-env-changed=DEEPFILTER_EMBED_MODEL");
    if let Ok(path) = std::env::var("DEEPFILTER_EMBED_MODEL") {
        println!("cargo:rerun-if-changed={path}");
    }
}
