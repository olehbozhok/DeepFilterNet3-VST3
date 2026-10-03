# AGENTS.md

Project instructions for coding agents working in this repository.

## Before changing code

- Read `PLANS.md` for the durable implementation and verification state, and
  use `UPDATE_PLANS.md` as the original requirements brief.
- Preserve the plugin name, CLAP ID `com.deepfilter.noise-reduction`, VST3
  class ID `DeepFilterNR001\0`, and parameter IDs `atten_lim` and `mix` unless
  the user explicitly authorizes an identity migration.
- Follow the change-specific verification and retry budgets in `PLANS.md`;
  stop once affected acceptance passes. Completed historical gates are not
  prerequisites for unrelated maintenance. Production source, manifest, or
  build-configuration changes invalidate affected release/manual-host evidence;
  test-only edits invalidate affected test evidence, while documentation-only
  edits do not invalidate binaries. Revise the plan before new acceptance work.

## DSP and build invariants

- Use nice-plug and nice-plug-xtask. Do not reintroduce nih-plug.
- Keep nice-plug, nice-plug-egui, and egui on compatible versions when updating
  `plugin/Cargo.toml` and `Cargo.lock`. The current editor requires Rust 1.95
  or later. Use `Plugin::activate`/`ActivateContext` and the concrete
  `Plugin::Editor` associated type for the current framework.
- Treat DeepFilterNet, ndarray, Tract, and rubato as a shared compatibility
  boundary; assess model APIs and resampling/latency behavior before upgrading
  them independently.
- Enable AT MOST one embedded model feature; a build with none is legal and
  deliberate. The four shapes:

  | build | what it carries |
  | :-- | :-- |
  | default | `model-ll`, DeepFilterNet's low-latency model |
  | `--no-default-features --features model-standard` | DeepFilterNet's standard model |
  | `--no-default-features --features model-shortwave` | `plugin/models/dfn3-shortwave-v1_onnx.tar.gz`, always that one |
  | `--no-default-features --features model-custom` | the archive `DEEPFILTER_EMBED_MODEL` names; the build fails if it names nothing |
  | `--no-default-features` | nothing; `DEEPFILTER_MODEL` is required at run time |

  Two at once is a compile error: the variants differ in lookahead, so a binary
  holding both could report the wrong latency to the host.
- `model-shortwave` IGNORES `DEEPFILTER_EMBED_MODEL` and warns that it did. The
  override lives in its own feature on purpose - when one feature quietly
  changed what it embedded, two people could run the same build command and get
  different weights with nothing in the command to say so.
- A build with no model must FAIL to construct an engine, with a message naming
  `DEEPFILTER_MODEL`. It must never fall back to another model: a plugin that
  silently passes audio through something nobody chose makes every listening
  judgement afterwards worthless. Note that `DfParams::default()` PANICS when
  DeepFilterNet itself was built without a model, which is why that call is
  compiled only into the shapes that have one.
- Keep `DfTract`, model reconstruction, and persistent rubato converters on
  the worker. The audio callback must not allocate, lock, wait, log, call the
  model, or call a resampler.
- Preserve timestamp and generation matching, per-channel latency-aligned dry
  fallback, and the single shared DSP path for real-time, buffered, and
  offline modes. Only Offline may wait, and its wait must remain bounded.
- Derive collection runway from the negotiated maximum host block plus one
  model quantum. Keep it in reported latency and both output timelines;
  deterministic immediate-worker tests alone do not prove real-time coverage.
- Advance parameter smoothing in audio-sample time. On input queue overflow,
  restart the worker generation without resetting host counters or dry delay,
  and keep aligned dry until the new model/resampler history becomes valid.
- Continue model advancement at an effectively zero attenuation setting while
  selecting the aligned raw path. Do not use DeepFilterNet's immediate
  zero-attenuation return as host output.
- Supported enhanced rates are 44.1, 48, 88.2, 96, 176.4, and 192 kHz.
  Unsupported rates, invalid layouts, queue limits, or startup failures must
  initialize as unchanged direct bypass with zero reported latency.

## Editor invariant

- Keep the custom editor fixed-size and English-only with exactly two
  interactive controls: the existing `atten_lim` and `mix` parameter sliders.
  Do not add model selection, waveforms, meters, bypass, presets, or other
  controls unless the user explicitly expands the UI scope.
- Keep GUI work outside the audio callback and route slider gestures through
  nice-plug's parameter setter so host automation remains synchronized.
- In `plugin/src/editor.rs`, preserve the `NiceEguiApp` lifecycle and the
  framework's `RepaintNotifier` integration so host automation repaints the UI.
  Reset temporary text-entry and drag state when the editor is reopened.

## Release packaging

- Keep the project license as MIT in `plugin/Cargo.toml` and root `LICENSE`.
  Treat files under `third-party-licenses/` and `THIRD_PARTY_NOTICES.md` as
  third-party terms, not alternative project licenses.
- Keep license documentation concise: do not restore dependency-purpose prose
  or license-expression tables. Preserve the embedded-model redistribution
  warning, required MIT/ISC/font notices, canonical Apache-2.0, Unicode, and
  embedded-font texts, and the exact VST trademark attribution in repository
  and package output.
- After a successful release bundle build, create the Apple Silicon archive
  only with `./scripts/package-release.sh`. The script must keep verifying thin
  arm64 binaries, valid ad-hoc signatures, ZIP integrity, and SHA-256 output;
  do not weaken its non-overwrite behavior or remove `ditto --norsrc`.
- Keep `githubreadme/screensho.png`, `githubreadme/effect-off.wav`, and
  `githubreadme/effect-on.wav` as repository README assets. Do not include
  image or audio assets in the release ZIP.
- Keep generated `dist/` artifacts untracked. Attach both the ZIP and its
  `.zip.sha256` sidecar when a release is eventually published.
