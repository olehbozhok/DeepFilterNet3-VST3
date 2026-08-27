"""Fine-tune DeepFilterNet3 on pre-mixed noisy/clean pairs.

WHY THIS EXISTS
---------------
Upstream DeepFilterNet trains by mixing speech and noise AT RUNTIME: its Rust
dataloader knows three dataset kinds - Speech, Noise, RIR - and builds every
example by drawing one speech clip, two to five noise clips, and an SNR. There
is no path for a corpus that is already paired.

That matters when the pairing carries information the runtime mixer cannot
reproduce. For shortwave that is at least: a slow fading envelope on the speech,
several interferers at independent levels, and a receiver passband applied to
BOTH sides of the pair - upstream's own bandwidth distortion is applied to the
noisy side alone, deliberately, because there it defines a bandwidth-extension
task rather than a denoising one.

So this trains the upstream model, with the upstream loss, on pre-mixed pairs.
Everything about the network and the objective comes from the installed
`deepfilternet` package; only the data path is ours.

WHAT IT DOES NOT DO
-------------------
It does not replace upstream training, it does not touch the plugin, and it
writes nothing outside the directory given to `--out-dir`. The base checkpoint
is opened read-only and never overwritten: it is the baseline every result is
measured against.

DATA LAYOUT EXPECTED
--------------------
    <data-dir>/
      manifest.csv          optional; if present it is authoritative
      train/clean/00000.wav
      train/noisy/00000_0.wav
      test/clean/...
      test/noisy/...

A noisy file belongs to the clean file whose stem is its stem up to the first
underscore, so one clean clip may have several noisy versions. With a manifest,
the pairing and the per-pair SNR are read from it instead of guessed.

ROBUSTNESS, AND WHY EACH PIECE IS THERE
---------------------------------------
Training runs for hours on a laptop GPU. Every one of these is about not losing
that time:

  a checkpoint is written every `--save-every-steps` steps AND at every epoch
  end, atomically - to a temporary file, then renamed - so an interruption
  during the write cannot leave a truncated checkpoint where the last good one
  used to be.

  a checkpoint carries the optimizer, the scaler, the epoch, the step and the
  RNG states, so resuming continues the run rather than starting a similar one.

  running out of VRAM halves the micro-batch and retries the same step instead
  of ending the run. Gradient accumulation keeps the effective batch size fixed,
  so the halving changes memory and not the optimization.

  Ctrl-C writes a checkpoint before exiting.

  `--check-loss` runs the base model over one batch and prints the loss without
  touching any weights. Use it before the first real run: this file computes
  features itself instead of receiving them from upstream's dataloader, and a
  mistake there would train something quietly different from what the model
  expects at inference.

Everything is a command-line argument. There are no paths in this file, and
it writes nothing outside `--out-dir`.

Usage:
    python training/finetune_dfn.py --data-dir DIR --model-base-dir DIR \\
        --out-dir DIR [--check-loss]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# The heavy dependencies are guarded rather than imported bare. torch, numpy and
# soundfile are genuine requirements - Dataset is a base class here, so deferring
# them into main() would contort the file - but a missing environment should
# produce one sentence saying what to install, not a ModuleNotFoundError from
# line eighty.
_ENV_HELP = (
    "the training environment is not set up in the interpreter running this "
    "script.\n"
    "It needs its own virtualenv - see training/README.md:\n"
    "    python -m venv training/.venv\n"
    "    <venv>/pip install torch --index-url "
    "https://download.pytorch.org/whl/cu128\n"
    "    <venv>/pip install -r training/requirements.txt"
)

try:
    import numpy as np
    import soundfile as sf
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as _exc:                                     # noqa: BLE001
    raise SystemExit(f"{_ENV_HELP}\n\n(missing: {_exc.name})") from None

# Upstream DeepFilterNet. Imported inside main() so a missing install produces
# one clear sentence at the point of use rather than a traceback three frames
# down, and so this file can be read and linted without it.
_DF_IMPORT_ERROR = (
    "the `deepfilternet` package is required and is not importable.\n"
    "Install it into the training environment, for example:\n"
    "    pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
    "    pip install deepfilternet\n"
    "See training/README.md for the version notes."
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    clean: Path
    noisy: Path
    snr_db: float = float("nan")
    cutoff_hz: float = float("nan")


def pairs_from_manifest(data_dir: Path, split: str) -> list[Pair]:
    """Pairs as the corpus builder recorded them.

    Preferred over globbing because it also carries the SNR, which the loss can
    use, and because it is the only place that knows which noisy file came from
    which clean one when the naming convention changes.
    """
    manifest = data_dir / "manifest.csv"
    out: list[Pair] = []
    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != split:
                continue
            clean = data_dir / split / "clean" / row["clean"]
            noisy = data_dir / split / "noisy" / row["noisy"]
            if not (clean.exists() and noisy.exists()):
                continue
            out.append(Pair(
                clean=clean, noisy=noisy,
                snr_db=float(row.get("snr_db", "nan") or "nan"),
                cutoff_hz=float(row.get("cutoff_hz", "nan") or "nan"),
            ))
    return out


def pairs_from_layout(data_dir: Path, split: str) -> list[Pair]:
    """Pairs recovered from the directory layout, when there is no manifest."""
    clean_dir, noisy_dir = data_dir / split / "clean", data_dir / split / "noisy"
    out: list[Pair] = []
    for noisy in sorted(noisy_dir.glob("*.wav")):
        clean = clean_dir / (noisy.stem.split("_", 1)[0] + ".wav")
        if clean.exists():
            out.append(Pair(clean=clean, noisy=noisy))
    return out


def load_pairs(data_dir: Path, split: str) -> list[Pair]:
    if (data_dir / "manifest.csv").exists():
        got = pairs_from_manifest(data_dir, split)
        if got:
            return got
    return pairs_from_layout(data_dir, split)


class PairDataset(Dataset):
    """Random fixed-length crops of pre-mixed pairs.

    The crop is taken at the SAME offset from both sides - they are two views of
    one moment and are meaningless apart - and it is random per epoch, so ten
    seconds of material is not one training example forever.
    """

    def __init__(self, pairs: list[Pair], sr: int, seg_samples: int,
                 train: bool, seed: int = 0, with_features: bool = False,
                 config_path: Path | None = None):
        self.pairs = pairs
        self.sr = sr
        self.seg = seg_samples
        self.train = train
        self.seed = seed
        self.with_features = with_features
        self.config_path = config_path
        self._state = None
        self._nb_df = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def _read(self, path: Path, start: int, n: int) -> np.ndarray:
        x, sr = sf.read(str(path), start=start, frames=n, dtype="float32",
                        always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != self.sr:
            raise RuntimeError(
                f"{path.name} is {sr} Hz, the model wants {self.sr} Hz. "
                f"Rebuild the corpus at the model's rate rather than resampling "
                f"here - a silent resample in the data path is exactly the kind "
                f"of difference that shows up as a bad result three days later.")
        if len(x) < n:
            x = np.pad(x, (0, n - len(x)))
        return x

    def _df_state(self):
        """One DF state per worker process, built on first use.

        The features used to be computed in the training loop: a Python loop
        over the batch, twice per step, on one core, while the GPU waited. They
        depend only on the audio, so they belong in the loader, where N workers
        compute them in parallel and overlap with the step before.

        The state is Rust-backed and cannot be pickled into a worker, so each
        worker builds its own. `config` is process-global upstream and refuses a
        second load, which is exactly right here: fresh in a worker, already
        loaded when num_workers is 0.
        """
        if self._state is not None:
            return self._state
        from df.config import config
        from df.model import ModelParams
        from libdf import DF

        try:
            config.load(str(self.config_path), allow_defaults=True)
        except ValueError:
            pass                     # already loaded in this process
        p = ModelParams()
        self._nb_df = p.nb_df
        self._state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size,
                         nb_bands=p.nb_erb, min_nb_erb_freqs=p.min_nb_freqs)
        return self._state

    def __getitem__(self, i: int):
        p = self.pairs[i]
        info = sf.info(str(p.noisy))
        total = int(info.frames)
        if self.train and total > self.seg:
            # Deterministic per (seed, index) so a resumed run sees the same
            # crops it would have seen, rather than a different dataset.
            rng = random.Random(self.seed * 1_000_003 + i)
            start = rng.randrange(0, total - self.seg)
        else:
            start = max(0, (total - self.seg) // 2)
        clean = self._read(p.clean, start, self.seg)
        noisy = self._read(p.noisy, start, self.seg)
        snr = float(p.snr_db if not math.isnan(p.snr_db) else 0.0)

        if not self.with_features:
            return torch.from_numpy(clean), torch.from_numpy(noisy), snr

        from df.enhance import df_features
        from df.utils import as_real

        state = self._df_state()
        spec_n, erb, feat = df_features(torch.from_numpy(noisy).unsqueeze(0),
                                        state, self._nb_df)
        # The clean side needs its spectrogram and nothing else - the loss reads
        # it, the model never sees it - so only the analysis is run, not the
        # whole feature set.
        spec_c = as_real(torch.as_tensor(
            state.analysis(clean[None, :])).unsqueeze(1))
        return (spec_c.squeeze(0), spec_n.squeeze(0), erb.squeeze(0),
                feat.squeeze(0), snr)


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, *, model, optimizer, scaler, epoch: int,
                    step: int, best: float, args_snapshot: dict) -> None:
    """Write a checkpoint that a resume can actually continue from.

    Atomic: written beside the target and renamed. A checkpoint half-written
    over the previous one is worse than no checkpoint, because it looks like a
    checkpoint.
    """
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "best": best,
        "args": args_snapshot,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": (torch.cuda.get_rng_state_all()
                     if torch.cuda.is_available() else None),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, *, model, optimizer=None, scaler=None) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ckpt.get("torch_rng") is not None:
        torch.set_rng_state(ckpt["torch_rng"].to(torch.uint8))
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        except Exception:                                       # noqa: BLE001
            pass
    if ckpt.get("numpy_rng") is not None:
        np.random.set_state(ckpt["numpy_rng"])
    if ckpt.get("python_rng") is not None:
        random.setstate(ckpt["python_rng"])
    return ckpt


def save_upstream_checkpoint(out_dir: Path, model, epoch: int) -> Path:
    """Write the weights the way upstream reads them.

    Two things need this and neither can read our format: the real-pair
    evaluation, which loads a candidate with `df.checkpoint.load_model`, and
    `df/scripts/export.py`, which is how a finished model reaches the plugin.
    `read_cp` wants a bare state_dict under `checkpoints/model_<epoch>.ckpt`, so
    that is what goes there - beside, not instead of, the resumable checkpoint
    that carries the optimizer and the RNG.
    """
    cp_dir = out_dir / "checkpoints"
    cp_dir.mkdir(parents=True, exist_ok=True)
    path = cp_dir / f"model_{epoch}.ckpt.best"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, tmp)
    os.replace(tmp, path)
    # read_cp takes the highest epoch it finds, so old ones would win nothing -
    # but they would still fill the disk over a long run.
    for old_cp in sorted(cp_dir.glob("model_*.ckpt.best")):
        if old_cp != path:
            old_cp.unlink(missing_ok=True)
    return path


def prune_checkpoints(out_dir: Path, keep: int) -> None:
    """Keep the newest `keep` step checkpoints. `best` and `last` are never
    pruned - they are the two a resume or an evaluation actually reaches for."""
    steps = sorted(out_dir.glob("step_*.pt"),
                   key=lambda p: int(p.stem.split("_")[1]))
    for old in steps[:-keep] if keep > 0 else []:
        old.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class OomPolicy:
    """How the run responds to running out of VRAM.

    Halving the micro-batch and raising accumulation keeps the EFFECTIVE batch
    constant, so the optimization does not change when memory pressure does.
    A run that quietly trains at a different batch size after an OOM would be a
    different experiment wearing the same name.
    """
    micro: int
    accum: int
    floor: int = 1
    events: list = field(default_factory=list)

    ceiling: int = 0          # never grow past the configured effective batch
    quiet: int = 0            # consecutive steps with room to spare

    def shrink(self, step: int) -> bool:
        if self.micro <= self.floor:
            return False
        self.micro = max(self.floor, self.micro // 2)
        self.accum *= 2
        self.quiet = 0
        self.events.append({"step": step, "micro": self.micro,
                            "accum": self.accum, "why": "out of memory"})
        return True

    def maybe_grow(self, step: int, peak_bytes: float, cap_bytes: float,
                   target: float = 0.75, patience: int = 30) -> bool:
        """Take the micro-batch back up while there is room to spare.

        It grows only up to the EFFECTIVE batch, and that bound is the whole
        design. The effective batch is a hyperparameter of the experiment: a run
        that quietly enlarged it because the card had room would be a different
        experiment wearing the same name, and its result would not be comparable
        to the one before it. What this recovers is accumulation - the micro-batch
        halved by an earlier out-of-memory event, or a conservative starting
        value - which changes speed and nothing else.

        `target` is deliberately well under the cap. Peak usage varies between
        steps with the content, so growing at 95% would mean growing straight
        into the next shrink, and each of those costs a discarded batch.
        """
        if self.ceiling and self.micro >= self.ceiling:
            return False
        if cap_bytes <= 0 or peak_bytes / cap_bytes > target:
            self.quiet = 0
            return False
        self.quiet += 1
        if self.quiet < patience:
            return False
        if self.accum <= 1:
            return False
        self.micro = min(self.ceiling or self.micro * 2, self.micro * 2)
        self.accum = max(1, self.accum // 2)
        self.quiet = 0
        self.events.append({"step": step, "micro": self.micro,
                            "accum": self.accum, "why": "room to spare"})
        return True


def build_features(df_state, nb_df: int, audio: torch.Tensor, device):
    """Waveform -> (spec, feat_erb, feat_spec), exactly as inference does it.

    Mirrors `df.enhance.df_features`. It has to: the features a fine-tune is
    trained on and the features the plugin computes at inference must be the
    same function, or the model is being trained for a different input than it
    will ever see.
    """
    from df.enhance import df_features as _df_features

    specs, erbs, sfeats = [], [], []
    for row in audio:
        spec, erb, sfeat = _df_features(row.unsqueeze(0).cpu(), df_state, nb_df)
        specs.append(spec)
        erbs.append(erb)
        sfeats.append(sfeat)
    spec = torch.cat(specs, dim=0).to(device)
    feat_erb = torch.cat(erbs, dim=0).to(device)
    feat_spec = torch.cat(sfeats, dim=0).to(device)
    return spec, feat_erb, feat_spec


class Progress:
    """A line every `every` seconds saying where the run is and when it ends.

    Not every step: at eighty steps a second that is a log nobody reads and a
    measurable slowdown. Not only at epoch end either - a run that says nothing
    for twenty minutes is indistinguishable from a hung one, which is a state
    this project has already mistaken for a working capture.
    """

    def __init__(self, every: float = 10.0):
        # every <= 0 silences it entirely, which is what --quiet sets.
        self.every = every
        self.last = 0.0
        self.t0 = time.time()
        self.stage = ""

    def show(self, stage: str, epoch: int, epochs: int, done: int, total: int,
             loss: float = float("nan"), force: bool = False) -> None:
        now = time.time()
        if (stage, epoch) != self.stage:
            # The clock restarts with the stage. Timing validation from the
            # start of the run made its rate the average of two different
            # things and its estimate nonsense - 18 minutes for fifty steps
            # that took nine seconds.
            self.stage = (stage, epoch)
            self.t0 = now
        if self.every <= 0:
            return                    # --quiet, and forced lines are silenced too
        if not force and now - self.last < self.every:
            return
        self.last = now
        elapsed = now - self.t0
        rate = done / elapsed if elapsed > 0 else 0.0
        left = (total - done) / rate if rate > 0 and total else float("inf")
        eta = ("--:--" if left != left or left in (float("inf"),) or left > 86400
               else f"{int(left) // 60:02d}:{int(left) % 60:02d}")
        bar = f"{done}/{total}" if total else str(done)
        msg = (f"[{stage:<8}] epoch {epoch + 1}/{epochs}  {bar:>13}  "
               f"{rate:5.2f} it/s  eta {eta}")
        if loss == loss:
            msg += f"  loss {loss:.4f}"
        print(msg, flush=True)


def freeze_parts(model, what: str) -> tuple[int, int]:
    """Freeze part of the network, and say how much was frozen.

    Full fine-tuning on a corpus this size risks catastrophic forgetting: the
    model becomes good at one broadcaster on shortwave and worse at everything
    it already handled. Freezing the encoder keeps the learned representation
    and adapts only what turns it into a filter.
    """
    if what == "none":
        pass
    elif what == "encoder":
        for name, param in model.named_parameters():
            if name.startswith("enc"):
                param.requires_grad_(False)
    elif what == "erb-decoder":
        for name, param in model.named_parameters():
            if name.startswith("erb_dec"):
                param.requires_grad_(False)
    elif what == "df-decoder":
        for name, param in model.named_parameters():
            if name.startswith("df_dec"):
                param.requires_grad_(False)
    else:
        raise SystemExit(f"unknown --freeze value: {what}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, type=Path,
                    help="corpus root holding train/ and test/ (and optionally "
                         "manifest.csv)")
    ap.add_argument("--model-base-dir", required=True, type=Path,
                    help="DeepFilterNet checkpoint directory - the one holding "
                         "config.ini and checkpoints/. Opened read-only")
    ap.add_argument("--out-dir", required=True, type=Path,
                    help="everything this run writes goes here and nowhere else")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="effective batch. Micro-batching and accumulation keep "
                         "it constant if memory forces a smaller step")
    ap.add_argument("--micro-batch", type=int, default=0,
                    help="samples per forward pass; 0 means start equal to "
                         "--batch-size and shrink only if VRAM demands it")
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="fine-tuning, so well below the 1e-3 upstream trains "
                         "from scratch with")
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--seg-seconds", type=float, default=3.0,
                    help="crop length; upstream's own max_sample_len_s is 3.0")
    ap.add_argument("--freeze", default="encoder",
                    choices=("none", "encoder", "erb-decoder", "df-decoder"))
    ap.add_argument("--device", default="")
    ap.add_argument("--workers", type=int, default=4,
                    help="loader processes. NOT the core count: each one is a "
                         "separate process on Windows that re-imports torch and "
                         "df, and the loader is not the bottleneck. Measured at "
                         "batch 8 - 100 steps take 23 s with 0 workers and 40 s "
                         "with 8, because the startup dominates; 300 steps take "
                         "56 s with 0 and 51 s with 4, because by then it does "
                         "not. Worth having for real epochs, not for smoke tests")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--quiet", action="store_true", default=None,
                    help="print only what a reader needs afterwards: the final "
                         "loss, the checkpoint, and anything that went wrong. "
                         "Defaults ON when stdout is not a terminal, because "
                         "then nobody is watching it scroll - a piped or "
                         "captured run pays for every line and reads none of "
                         "them. --no-quiet forces the full output back")
    ap.add_argument("--no-quiet", dest="quiet", action="store_false")
    ap.add_argument("--progress-seconds", type=float, default=10.0,
                    help="how often to print a progress line; 0 prints "
                         "only at the end of each stage")
    ap.add_argument("--save-every-steps", type=int, default=200)
    ap.add_argument("--keep-last", type=int, default=3,
                    help="step checkpoints to keep; best and last are always kept")
    ap.add_argument("--max-steps-per-epoch", type=int, default=0,
                    help="0 = the whole set. A bound is useful on a laptop, "
                         "where a shorter epoch that finishes beats a long one "
                         "that thermal-throttles")
    ap.add_argument("--resume", default="auto",
                    help="'auto' continues from <out-dir>/last.pt if it exists, "
                         "'none' starts fresh, or a path to a checkpoint")
    ap.add_argument("--vram-fraction", type=float, default=0.85,
                    help="cap the share of VRAM this process may allocate. On "
                         "Windows the driver does NOT fail when VRAM runs out - "
                         "it silently spills into system RAM over PCIe, which "
                         "does not raise, does not log, and can take the machine "
                         "into swap while looking like a merely slow run. The "
                         "cap turns that into a real OutOfMemoryError, which is "
                         "what the batch halving is waiting for. 0 disables it")
    ap.add_argument("--amp", action="store_true",
                    help="mixed precision. Off by default: it halves memory and "
                         "it also changes the numerics, and this model's loss "
                         "works on spectra where that has not been checked here")
    ap.add_argument("--check-loss", action="store_true",
                    help="run the base model over one batch, print the loss, "
                         "change nothing, exit")
    args = ap.parse_args()
    if args.quiet is None:
        args.quiet = not sys.stdout.isatty()
    if args.quiet:
        args.progress_seconds = 0.0
        # The df package logs its own INFO lines through loguru, and they are
        # most of the noise: a model init and a checkpoint path per run.
        try:
            from loguru import logger

            logger.remove()
            logger.add(sys.stderr, level="WARNING")
        except Exception:                                       # noqa: BLE001
            pass

    def say(*a, **kw):
        """Progress talk. Silent when nobody is reading."""
        if not args.quiet:
            say(*a, **kw)

    try:
        from df.checkpoint import load_model
        from df.config import config
        from df.loss import Istft, Loss
        from df.model import ModelParams
        from df.utils import detach_hidden
        from libdf import DF
    except ImportError as exc:                                  # noqa: BLE001
        print(f"{_DF_IMPORT_ERROR}\n\n({exc})", file=sys.stderr)
        return 2

    cfg = args.model_base_dir / "config.ini"
    if not cfg.exists():
        say(f"no config.ini in {args.model_base_dir} - point --model-base-dir "
              f"at a DeepFilterNet checkpoint directory", file=sys.stderr)
        return 2
    config.load(str(cfg), allow_defaults=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and args.vram_fraction > 0:
        # Without this a run does not fail when it runs out of VRAM: it gets
        # slower, the machine starts swapping, and nothing says so.
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        total = torch.cuda.get_device_properties(0).total_memory
        say(f"VRAM cap {args.vram_fraction:.0%} of {total / 1e9:.1f} GB = "
              f"{args.vram_fraction * total / 1e9:.1f} GB - past it the run "
              f"raises instead of spilling into system RAM")
    p = ModelParams()
    df_state = DF(sr=p.sr, fft_size=p.fft_size, hop_size=p.hop_size,
                  nb_bands=p.nb_erb, min_nb_erb_freqs=p.min_nb_freqs)

    # The base checkpoint is READ. Nothing this script does writes into
    # --model-base-dir: it is the baseline, and a baseline that can be modified
    # by the thing it is meant to judge is not one.
    #
    # `read_cp` globs `model*.ckpt` in the directory it is GIVEN, and a released
    # DeepFilterNet model puts them in a `checkpoints/` subdirectory next to
    # config.ini. Handed the parent, it finds nothing, returns None, and
    # `load_model` turns that into epoch 0 - a randomly initialised network, with
    # a log line that looks like success.
    #
    # That is not hypothetical. It happened here on 2026-08-27 and was caught
    # only by comparing the Python path against the reference binary: on a clip
    # at +9.9 dB SNR the untrained network returned -48.5 dBFS where the binary
    # returned -18.7. Fine-tuning would have run for hours, from noise, and the
    # loss would have gone down the whole time.
    cp_dir = args.model_base_dir / "checkpoints"
    if not cp_dir.is_dir():
        cp_dir = args.model_base_dir
    model, base_epoch = load_model(str(cp_dir), df_state, epoch="best")
    if base_epoch == 0:
        print(f"no checkpoint was loaded from {cp_dir}.", file=sys.stderr)
        say("`read_cp` looks for model*.ckpt or model*.ckpt.best there. "
              "Without one the network is randomly initialised, and "
              "training would start from noise while reporting nothing "
              "wrong.", file=sys.stderr)
        return 2
    model = model.to(device)

    trainable, total = freeze_parts(model, args.freeze)
    say(f"model: {total/1e6:.2f} M parameters, {trainable/1e6:.2f} M trainable "
          f"(--freeze {args.freeze}), base epoch {base_epoch}, device {device}")

    istft = Istft(p.fft_size, p.hop_size,
                  torch.as_tensor(df_state.fft_window().copy())).to(device)
    losses = Loss(df_state, istft).to(device)

    seg = int(round(args.seg_seconds * p.sr))
    train_pairs = load_pairs(args.data_dir, "train")
    valid_pairs = load_pairs(args.data_dir, "test")
    if not train_pairs:
        print(f"no training pairs under {args.data_dir}", file=sys.stderr)
        return 2
    say(f"data: {len(train_pairs)} train pair(s), {len(valid_pairs)} held out, "
          f"{args.seg_seconds:.1f} s crops at {p.sr} Hz")

    micro = args.micro_batch or args.batch_size
    accum = max(1, args.batch_size // max(1, micro))
    oom = OomPolicy(micro=micro, accum=accum, ceiling=args.batch_size)
    vram_cap = 0.0
    if device.type == "cuda" and args.vram_fraction > 0:
        vram_cap = args.vram_fraction * torch.cuda.get_device_properties(0).total_memory

    def make_loader(pairs, train: bool, batch: int, epoch: int = 0):
        ds = PairDataset(pairs, p.sr, seg, train, seed=args.seed + epoch,
                         with_features=True, config_path=cfg)
        return DataLoader(ds, batch_size=batch, shuffle=train,
                          num_workers=args.workers, drop_last=train,
                          pin_memory=(device.type == "cuda"),
                          persistent_workers=args.workers > 0,
                          prefetch_factor=4 if args.workers > 0 else None)

    def run_batch(item):
        """One step's forward, on features the loader already computed."""
        spec_clean, spec_noisy, feat_erb, feat_spec, snrs = item
        spec_clean = spec_clean.to(device, non_blocking=True)
        spec_noisy = spec_noisy.to(device, non_blocking=True)
        feat_erb = feat_erb.to(device, non_blocking=True)
        feat_spec = feat_spec.to(device, non_blocking=True)
        enh, m, lsnr, _ = model.forward(spec=spec_noisy.clone(),
                                        feat_erb=feat_erb, feat_spec=feat_spec)
        # The loss wants the REAL-VALUED spectrogram form, [B, C, T, F, 2], not
        # complex: `df.modules.local_snr` asserts `clean.dim() == 5`. That is
        # what `df_features` already returns, so the tensors go in untouched.
        # Converting them to complex first raises an assertion four frames down
        # with no hint of which argument was wrong.
        err = losses.forward(spec_clean, spec_noisy, enh, m, lsnr,
                             snrs=snrs.to(device))
        return err

    if args.check_loss:
        # No optimizer, no gradients, nothing written. This exists because the
        # features are computed here rather than by upstream's dataloader, and
        # an error there would train a model for inputs it will never see.
        model.eval()
        loader = make_loader(train_pairs, False, min(4, len(train_pairs)))
        item = next(iter(loader))
        with torch.no_grad():
            err = run_batch(item)
        say()
        print(f"base model, one batch of {len(item[0])}: loss {float(err):.6f}")
        say("Nothing was written and no weight was changed.")
        say("Sanity to apply before trusting a training run:")
        say("  - the loss is finite and not absurd (order 1e-2..1e1 here)")
        say("  - it is LOWER on the held-out set than on random noise pairs")
        say("  - it does not change between two runs with the same seed")
        return 0

    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr,
                                  weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type) if args.amp else None

    args.out_dir.mkdir(parents=True, exist_ok=True)
    snapshot = {k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()}
    (args.out_dir / "run.json").write_text(
        json.dumps({"args": snapshot, "base_epoch": base_epoch,
                    "trainable": trainable, "total": total}, indent=2),
        encoding="utf-8")

    start_epoch, step, best = 0, 0, float("inf")
    resume_path = None
    if args.resume == "auto":
        cand = args.out_dir / "last.pt"
        resume_path = cand if cand.exists() else None
    elif args.resume != "none":
        resume_path = Path(args.resume)
    if resume_path is not None:
        ck = load_checkpoint(resume_path, model=model, optimizer=optimizer,
                             scaler=scaler)
        start_epoch, step = int(ck.get("epoch", 0)), int(ck.get("step", 0))
        best = float(ck.get("best", float("inf")))
        say(f"resumed from {resume_path} at epoch {start_epoch}, step {step}, "
              f"best {best:.6f}")

    log_path = args.out_dir / "train_log.csv"
    if not log_path.exists():
        with log_path.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(
                ["wall_s", "epoch", "step", "train_loss", "valid_loss",
                 "micro_batch", "accum", "lr"])

    stopping = {"now": False}

    def on_signal(signum, _frame):
        # One checkpoint, then leave. A run killed between the notice and the
        # write would lose the epoch, and the whole point of this block is that
        # hours of GPU time survive an interruption.
        say(f"\nsignal {signum} - writing a checkpoint before exiting")
        stopping["now"] = True

    signal.signal(signal.SIGINT, on_signal)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, on_signal)

    prog = Progress(args.progress_seconds)
    # An epoch is one pass over the clips, so the number of optimizer steps
    # is set by the EFFECTIVE batch, not by whatever the micro-batch happens
    # to be after a resize.
    steps_per_epoch = max(1, len(train_pairs) // max(1, args.batch_size))
    if args.max_steps_per_epoch:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    say(f"one epoch is about {steps_per_epoch} optimizer step(s) over "
          f"{len(train_pairs)} pair(s)")

    t0 = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running, seen, micro_i = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        epoch_start_step = step
        # Changing the micro-batch means a new DataLoader, and a new DataLoader
        # means starting the pass again. The first version simply broke out of
        # the loop, so every resize - an out-of-memory event or a growth - ended
        # the epoch early while reporting a normal epoch. `resized` is what
        # tells the difference between "the data ran out" and "the batch
        # changed under us".
        resized = True
        while resized and not stopping["now"]:
            resized = False
            loader = make_loader(train_pairs, True, oom.micro, epoch)
            for item in loader:
                try:
                    if scaler is not None:
                        with torch.amp.autocast(device.type):
                            err = run_batch(item) / oom.accum
                        scaler.scale(err).backward()
                    else:
                        err = run_batch(item) / oom.accum
                        err.backward()
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    if not oom.shrink(step):
                        say("out of memory at the smallest micro-batch - stopping "
                              "and saving", file=sys.stderr)
                        stopping["now"] = True
                        break
                    say(f"  out of memory: micro-batch -> {oom.micro}, "
                          f"accumulation -> {oom.accum}; the effective batch is "
                          f"unchanged", file=sys.stderr)
                    resized = True
                    break        # rebuilt below; the pass continues

                running += float(err.detach()) * oom.accum
                seen += 1
                micro_i += 1
                if micro_i % oom.accum == 0:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    # Cut the graph between steps, as upstream's own loop does. The
                    # network is recurrent; without this any hidden state kept on the
                    # module stays attached to the previous step's graph, which grows
                    # the backward pass without bound and makes the gradients wrong
                    # in a way that shows up as memory, not as an error.
                    detach_hidden(model)
                    step += 1

                    # Take the micro-batch back up when the card has room. Peak is
                    # reset each step so the reading is this step's, not the run's
                    # high-water mark - otherwise one heavy batch early on would
                    # keep the run small for ever.
                    if vram_cap and oom.maybe_grow(
                            step, torch.cuda.max_memory_allocated(), vram_cap):
                        say(f"  room to spare: micro-batch -> {oom.micro}, "
                              f"accumulation -> {oom.accum}; the effective batch "
                              f"is unchanged", flush=True)
                        torch.cuda.reset_peak_memory_stats()
                        resized = True
                        break        # rebuilt below, and the pass continues
                    if vram_cap and step % 20 == 0:
                        torch.cuda.reset_peak_memory_stats()

                    if args.save_every_steps and step % args.save_every_steps == 0:
                        save_checkpoint(args.out_dir / f"step_{step:07d}.pt",
                                        model=model, optimizer=optimizer,
                                        scaler=scaler, epoch=epoch, step=step,
                                        best=best, args_snapshot=snapshot)
                        save_checkpoint(args.out_dir / "last.pt", model=model,
                                        optimizer=optimizer, scaler=scaler,
                                        epoch=epoch, step=step, best=best,
                                        args_snapshot=snapshot)
                        prune_checkpoints(args.out_dir, args.keep_last)
                        say(f"  epoch {epoch} step {step} "
                              f"train {running/max(1,seen):.5f}", flush=True)

                    prog.show("train", epoch, args.epochs,
                              step - epoch_start_step, steps_per_epoch,
                              running / max(1, seen))

                    if args.max_steps_per_epoch and \
                            step - epoch_start_step >= args.max_steps_per_epoch:
                        break
                if stopping["now"]:
                    break

        # ---- held-out loss, for the checkpoint selection only ---------------
        valid = float("nan")
        if valid_pairs and not stopping["now"]:
            model.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                n_valid = min(50, max(1, len(valid_pairs) // max(1, oom.micro)))
                prog.show("validate", epoch, args.epochs, 0, n_valid, force=True)
                for item in make_loader(valid_pairs, False, oom.micro):
                    vs += float(run_batch(item))
                    vn += 1
                    prog.show("validate", epoch, args.epochs, vn, n_valid,
                              vs / max(1, vn))
                    if vn >= 50:      # enough to rank checkpoints, not a result
                        break
            valid = vs / max(1, vn)

        prog.show("train", epoch, args.epochs, step - epoch_start_step,
                  steps_per_epoch, running / max(1, seen), force=True)
        train_loss = running / max(1, seen)
        with log_path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(
                [round(time.time() - t0, 1), epoch, step, round(train_loss, 6),
                 round(valid, 6) if valid == valid else "",
                 oom.micro, oom.accum, args.lr])
        print(f"epoch {epoch}: train {train_loss:.5f}  held-out {valid:.5f}",
              flush=True)

        save_checkpoint(args.out_dir / "last.pt", model=model,
                        optimizer=optimizer, scaler=scaler, epoch=epoch + 1,
                        step=step, best=best, args_snapshot=snapshot)
        if valid == valid and valid < best:
            best = valid
            save_checkpoint(args.out_dir / "best.pt", model=model,
                            optimizer=optimizer, scaler=scaler, epoch=epoch + 1,
                            step=step, best=best, args_snapshot=snapshot)
            cp = save_upstream_checkpoint(args.out_dir, model, epoch + 1)
            print(f"  new best; upstream-format weights at {cp.name}")

        if stopping["now"]:
            say("stopped early; last.pt holds the current state")
            break

    if device.type == "cuda":
        say()
        print(f"peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")

    if oom.events:
        print("\nmemory pressure during this run:")
        for e in oom.events:
            print(f"  step {e['step']}: micro-batch {e['micro']}, "
                  f"accumulation {e['accum']}")

    print(f"\ncheckpoints in {args.out_dir}")
    if not args.quiet:
        print("The held-out loss ranks checkpoints. It does NOT say the model "
              "is better on the air: score the result on real recordings "
              "before believing anything about it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
