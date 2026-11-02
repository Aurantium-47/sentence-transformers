# Reproducible CPU/DDP evidence for synchronized per-dataset batches

This is a standalone, sanitized evidence package, separate from the seven-file
Sentence Transformers PR. The recorded runs
came from an independent validation environment; they are not newly run DDP jobs
from PR packaging, upstream CI, or a human maintainer endorsement.

## What is included

- Both original native Trainer harnesses (standard A32/B8 and supplementary
  A65/B9), their native-objective oracles and RR/PROPORTIONAL options.
- All **36 rank JSON traces** from **18 real two-process CPU/Gloo Trainer jobs**:
  standard RR/PROPORTIONAL at accumulation G=1 and G=2, tail RR/PROPORTIONAL at
  G=2, each with continuous two-epoch, prefix/save, and fresh-process resume.
- Four original validation summaries, the 18-job matrix summary, all 303 expected
  runtime package-file hashes, original dependency freeze, and exact production
  patch. No checkpoint, pickle, safetensors, binary traces or private log dump.
- A JSON-only oracle replay entrypoint and a bounded fresh-run launcher.

`traces/standard/` corresponds to the source `native_runs/`; `traces/tail/`
corresponds to `native_tail_runs/`. `SANITIZATION.json` maps every imported file,
records original and published SHA256s, and records each changed JSON pointer.
Only absolute machine-path strings were rewritten; JSON was compacted. Every
numeric value, sample, loss, gradient, parameter, momentum, return code and array
order is unchanged. The two harnesses have unchanged executable ASTs: only their
obsolete import-overlay instruction docstrings were replaced. Original evidence
is preserved elsewhere; no source artifact was overwritten.

## Install the exact runtime in an isolated environment

Commands below are run from this directory on Linux with Python 3.12 (the recorded
run used Python 3.12.14). Network access is needed for installation only; fixtures
and validation are offline. Use a Python 3.12 virtual environment; do not upgrade a
user machine's global packages. Do not use arbitrary untrusted checkpoint files.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
mkdir sources
git clone https://github.com/huggingface/sentence-transformers.git sources/sentence-transformers
git -C sources/sentence-transformers checkout --detach 6994c94f39a3ff30f21075af1eca46d3ac5e8a8d
git -C sources/sentence-transformers apply ../../sentence_transformers_production.patch
git clone https://github.com/huggingface/accelerate.git sources/accelerate
git -C sources/accelerate fetch origin pull/4491/head
git -C sources/accelerate checkout --detach af29ce4975100abf803dec0f195edaf61e3e7e5e
python -m pip install --no-deps -e sources/sentence-transformers -e sources/accelerate
python verify_runtime.py
```

Expected runtime: torch `2.9.0+cpu`, Transformers `5.0.0`, Datasets `4.4.1`.
The original independent run used Accelerate base
`f47439c1fb24721ea71ab98f9b512726b5a97828` plus the two nullable-metadata edits.
PR #4491 head `af29ce4975100abf803dec0f195edaf61e3e7e5e` has that base as its
parent, and its two production-file SHA256s equal the recorded originals exactly.
`verify_runtime.py` checks all **303 recorded runtime package files**, not just a
version string. No released Accelerate compatibility is implied by these commands.
No runtime import overlay or modified loss/Trainer-loop source is needed.

## Replay the recorded numerical evidence without checkpoints

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
python replay_oracle.py --output replayed_oracle.json
```

This invokes the original harness `verify_group` functions on JSON only. It
recomputes six native-objective oracle groups and checks observed tasks/shards,
row coverage, positive pairs, labels, losses, full gradients, parameters, SGD
momentum, second epochs, and continuous-versus-prefix/resume traces. It does not
load binary training state and does not claim to re-audit the omitted checkpoint
file inventory. The JSON-only replay was actually rerun while preparing this
package; `evidence/public_trace_oracle_replay.json` contains that fresh result.

## Rerun all 18 real DDP jobs

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python run_matrix.py --output fresh_runs
```

`fresh_runs` must not exist. The wrapper runs the standard harness twice (six
jobs each) and the tail harness twice (three jobs each), with a 660/360-second
outer deadline per strategy, a 90-second inner launcher deadline, and a
15-second Gloo timeout. On outer timeout/failure it kills tracked descendants (including the original
harness's separately-sessioned torchrun children) and the outer process group.
It polls the process tree every 0.1 seconds; the original harness also cleans up
timed-out worker descendants. A lightweight synthetic nested-session timeout
test was executed successfully: both descendants stopped, with none left running.
Fresh execution generates its own local checkpoints for actual resume, logs,
JSON traces and summaries. Those new checkpoints are not part of this published
package. The outer `run_matrix.py` wrapper is newly added packaging tooling, not the
launcher that produced the historical traces. It was syntax-checked and its
cleanup logic was exercised on synthetic processes, but **no DDP run was executed
with it during packaging**; the included 18 historical jobs are the original independent runs.
The original commands' path metadata is sanitized in the matrix summary; this
portable wrapper supplies the corresponding runnable commands without those paths.

## Numerical objective and supported boundary

Native gathered CachedMNRL is the global-query mean. Native CoSENT is the DDP
mean of rank-local CoSENT losses, **not CoSENT on a concatenated global batch**.
An accumulation window averages its microstep objectives rather than computing
a loss on pooled examples. No loss scaling or cross-rank CoSENT comparisons were
added. The standard fixture has 640 trainable parameters; the tail fixture has
1,184. All source hashes are exposed in the matrix/runtime metadata.

The recorded cases use W=2, per-rank batches 8/2, workers=0, ordinary random
sampling, explicit global dropping and complete accumulation windows. The tail
case establishes RR retained counts 2:2 and PROPORTIONAL 4:2, with nonzero dropping.
The largest reported absolute errors are below 2e-6; rank state and resumed-state
differences are zero in this deterministic fixture. No GPU/NCCL, mixed-precision,
arbitrary stochastic-transform/dropout continuation, speed/memory improvement,
or broad dependency-version support is asserted. NO_DUPLICATES/global text
deduplication and custom sampler modes are outside the feature's supported scope.

## Integrity and attribution

`SHA256SUMS` covers the final package files. `SANITIZATION.json` identifies the
source and transformation of imported evidence; `VALIDATION.json` describes the
checks actually performed during packaging. Source code remains subject to the
included upstream Apache 2.0 licenses. Significant AI assistance was used in the
feature, validation harnesses, analysis, sanitation and packaging; no user-personal
review or maintainer approval is asserted. This is reproducibility material for
review, not a substitute for upstream CI or the unreleased dependency's review.
