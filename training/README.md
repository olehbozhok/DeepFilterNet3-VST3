# training/

Fine-tuning DeepFilterNet3 on pre-mixed noisy/clean pairs.

This directory is **not part of the plugin build**. Nothing in `plugin/`,
`xtask/` or the Cargo workspace refers to it, no Rust target depends on it, and
`scripts/package-release.sh` does not include it. It is here because the model
the plugin embeds and the model this trains are the same model, and keeping them
in one repository is what stops the two drifting apart unrecorded.

## Why not upstream's trainer

Upstream DeepFilterNet trains by mixing speech and noise **at runtime**. Its
Rust dataloader knows three dataset kinds — Speech, Noise, RIR — and builds each
example from one speech clip, two to five noise clips and a sampled SNR. There
is no path for a corpus that is already paired.

For shortwave the pairing carries what the runtime mixer cannot reproduce:

- a slow fading envelope on the speech,
- several interferers at independent levels,
- a receiver passband applied to **both** sides of the pair.

That last one is not a detail. Upstream applies its bandwidth distortion to the
noisy side alone, deliberately — there it defines a bandwidth-*extension* task.
Applied to one side in a denoising corpus it lets the model take a large part of
the task by learning "remove everything above the cutoff", because above the
cutoff the target is empty by construction and the input never is.

So: upstream's model, upstream's loss, our data path.

## Environment

Separate from any other environment in this repository or the corpus project.
It pulls in PyTorch and CUDA, which is several gigabytes and has no business
near the plugin build or the capture tooling.

```bash
python -m venv training/.venv
training/.venv/Scripts/pip install -r training/requirements.txt
```

`requirements.txt` deliberately does not pin a torch build: the right wheel
depends on the GPU. For an NVIDIA Blackwell card (RTX 50-series, compute
capability 12.0) it must be a CUDA 12.8 or newer build — PyTorch added that
architecture in 2.7:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

Check it before anything else, because the failure is not always loud:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_arch_list())
```

`sm_120` must appear in that list for a Blackwell card. If it does not, the
wheel is wrong and the run will either fall back to the CPU or fail on the first
kernel.

## Data layout

```
<data-dir>/
  manifest.csv          optional; authoritative when present
  train/clean/00000.wav
  train/noisy/00000_0.wav
  test/clean/...
  test/noisy/...
```

A noisy file belongs to the clean file whose stem is its stem up to the first
underscore, so one clean clip may carry several noisy versions. With a manifest
the pairing and the per-pair SNR are read rather than guessed.

**The corpus must already be at the model's sample rate.** The script refuses to
resample: a silent resample in the data path is the kind of difference that
surfaces as a bad result days later, with nothing pointing at the cause.

## Running it

Check the loss path first. Features are computed here rather than by upstream's
dataloader, and a mistake there trains the model for inputs it will never see:

```bash
python training/finetune_dfn.py \
    --data-dir <corpus> --model-base-dir <dfn-checkpoint> \
    --out-dir <run-dir> --check-loss
```

That runs the base model over one batch, prints the loss, writes nothing and
changes no weight. Then:

```bash
python training/finetune_dfn.py \
    --data-dir <corpus> --model-base-dir <dfn-checkpoint> \
    --out-dir <run-dir> --epochs 10 --batch-size 8 --freeze encoder
```

Nothing is written outside `--out-dir`. `--model-base-dir` is opened read-only:
it is the baseline every result is measured against, and a baseline the
experiment can modify is not one.

## Surviving a long run

Every one of these is about not losing hours of GPU time:

| | |
| :-- | :-- |
| checkpoint cadence | every `--save-every-steps` steps and at every epoch end |
| atomicity | written to a temporary file and renamed, so an interrupted write cannot replace the last good checkpoint with a truncated one |
| contents | model, optimizer, scaler, epoch, step and RNG states — a resume continues the run rather than starting a similar one |
| out of memory | halves the micro-batch and doubles accumulation, keeping the **effective** batch fixed, and carries on. A run that quietly trains at a different batch size after an OOM is a different experiment wearing the same name |
| Ctrl-C | writes a checkpoint before exiting |
| retention | newest `--keep-last` step checkpoints, plus `best.pt` and `last.pt`, which are never pruned |

On a laptop, `--max-steps-per-epoch` is worth using: a shorter epoch that
finishes beats a long one that thermal-throttles halfway.

## Getting a trained model into the plugin

`df/scripts/export.py` from the upstream checkout turns a checkpoint into the
three ONNX graphs and the `config.ini` the plugin's Rust side loads, tarred as
`<name>_onnx.tar.gz`:

```bash
python df/scripts/export.py --model-base-dir <run-dir> <export-dir>
```

Check it before trusting it. `deep-filter.exe -m <archive>` runs the exported
model through the same tract path the plugin uses, so the export can be compared
against its own PyTorch original without touching the plugin at all. Ours agreed
at correlation 0.989-0.993 with zero lag and levels within 0.3 dB; the residual
is tract's operation order, not a difference in weights.

The plugin can take it three ways.

**At run time**, without rebuilding - the loop for comparing models by ear:

```bash
DEEPFILTER_MODEL=<path-to>_onnx.tar.gz   # then start the host
```

**Compiled in**, for shipping one:

```bash
cargo build --release -p deepfilter-vst   --no-default-features --features model-shortwave
```

That embeds `plugin/models/dfn3-shortwave-v1_onnx.tar.gz`, which is in this
repository for the same reason DeepFilterNet keeps its own models in its - the
model that ships is versioned with the code that ships it, and the build needs
nothing from the environment to be reproducible. Ours is 8.0 MB; the one cargo
already pulls in from upstream is 34.7 MB.

To compile in a different one - a new candidate, before it is adopted - use the
OTHER feature:

```bash
DEEPFILTER_EMBED_MODEL=<path-to>_onnx.tar.gz   cargo build --release -p deepfilter-vst   --no-default-features --features model-custom
```

The override has its own feature so that the build command always says which
model it produces. When it was a hidden switch on `model-shortwave`, two people
running the same command could get different weights with nothing to show it -
`model-shortwave` now ignores the variable and warns that it did.

`build.rs` resolves the path, because `include_bytes!` needs a literal and
cannot make that choice, and declares both the variable and the resolved file as
build inputs - so retraining the model rebuilds the plugin instead of leaving
stale weights behind a fresh path. A missing archive stops the build rather than
falling back to another model.

**With no model at all**, which then requires `DEEPFILTER_MODEL`:

```bash
cargo build --release -p deepfilter-vst --no-default-features
```

Two embedded models at once is a compile error. A build with none refuses to
construct an engine and says so; it never falls back to another model, because a
plugin that quietly passes audio through something nobody chose makes every
listening judgement after it worthless.

An environment variable rather than a control in the editor, because the editor
is specified to carry exactly two parameter sliders and a model chooser is not a
mix control. A missing or unreadable file is an error, not a silent fallback to
the embedded model: someone who set the variable wants that model, and quietly
running a different one would make every measurement afterwards a lie about
which model produced it.

## What "better" means

The held-out loss ranks checkpoints. It does **not** say the model is better on
the air — it is computed on the same simulation the model was trained on, and
that simulation is a hypothesis, not the ionosphere.

Score the result on real verified recordings before believing anything about it.
In the corpus project that is `scripts/eval_on_real_pairs.py`, which runs the
base model and a candidate over identical real windows and compares three
things: speech level change, syllabic correlation against the broadcaster's own
feed, and the residual noise left in the pauses.
