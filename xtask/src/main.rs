//! Bundling guard around `nice-plug-xtask`.
//!
//! The `plugin` feature is not in the default set, so a plain `cargo xtask
//! bundle deepfilter-vst` builds the core library, finds no plugin exports, and
//! - left to nice-plug-xtask alone - prints "Not creating any plugin bundles"
//! and exits 0. Stale bundles from an earlier run would then be what a release
//! script packages. Fail before building instead.

fn main() -> nice_plug_xtask::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bundle = args
        .first()
        .is_some_and(|command| matches!(command.as_str(), "bundle" | "bundle-universal"));
    if bundle && !plugin_requested(&args) {
        eprintln!(
            "error: `cargo xtask {}` needs the `plugin` feature, for example\n  \
             cargo xtask bundle deepfilter-vst --release --features plugin\n\
             Without it the build is the core library, which exports no plugin: \
             the bundle step would succeed while producing nothing.",
            args[0]
        );
        std::process::exit(2);
    }
    nice_plug_xtask::main()
}

/// Whether the arguments turn on the `plugin` feature.
///
/// Handles `--features plugin`, `--features=plugin`, comma- or space-separated
/// lists, and `--all-features` (which enables `plugin` along with the model
/// features the compiler's own exclusivity guard rejects).
fn plugin_requested(args: &[String]) -> bool {
    let mut index = 1;
    while index < args.len() {
        let arg = &args[index];
        if arg == "--all-features" {
            return true;
        }
        if arg == "--features" {
            index += 1;
            while index < args.len() && !args[index].starts_with('-') {
                if features_include_plugin(&args[index]) {
                    return true;
                }
                index += 1;
            }
            continue;
        }
        if let Some(list) = arg.strip_prefix("--features=") {
            if features_include_plugin(list) {
                return true;
            }
        }
        index += 1;
    }
    false
}

fn features_include_plugin(list: &str) -> bool {
    list.split([',', ' '])
        .any(|feature| feature == "plugin")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn plugin_is_found_in_every_spelling_cargo_accepts() {
        assert!(plugin_requested(&args(&["bundle", "p", "--features", "plugin"])));
        assert!(plugin_requested(&args(&["bundle", "p", "--features=plugin"])));
        assert!(plugin_requested(&args(&["bundle", "p", "--features", "plugin,model-ll"])));
        assert!(plugin_requested(&args(&["bundle", "p", "--features", "plugin model-ll"])));
        assert!(plugin_requested(&args(&[
            "bundle", "p", "--features", "model-ll", "plugin"
        ])));
        assert!(plugin_requested(&args(&["bundle-universal", "p", "--all-features"])));
    }

    #[test]
    fn a_core_only_bundle_is_rejected() {
        assert!(!plugin_requested(&args(&["bundle", "deepfilter-vst", "--release"])));
        assert!(!plugin_requested(&args(&[
            "bundle", "deepfilter-vst", "--no-default-features", "--features", "model-standard"
        ])));
        assert!(!plugin_requested(&args(&[
            "bundle", "deepfilter-vst", "--features", "nice-plug/assert_process_allocs"
        ])));
        assert!(!plugin_requested(&args(&["known-packages"])));
    }

    #[test]
    fn the_allocation_assertion_command_passes_with_plugin() {
        assert!(plugin_requested(&args(&[
            "bundle",
            "deepfilter-vst",
            "--features",
            "plugin,nice-plug/assert_process_allocs"
        ])));
    }
}
