# GPU utilization: are we using the full GPU?

**Measured answer: no — a single Atari run uses only ~11–22% SM and ~0.7–1.0 GB
of 46–80 GB (~1–2% memory).** (`nvidia-smi pmon` on jobs 9586027/9586028.) For
reference, the LLM prompt-opt jobs on the same cluster run at ~100%.

## Why
This actor-only Atari workload is **CPU/env-bound**, not compute-bound. Each
rollout timestep blocks on `num_envs` ALE + cv2 `env.step`s (frame-skip 4,
grayscale, resize) across subprocesses; the Nature-CNN forward on a batch of
`num_envs=16` is trivial for an L40S/H100. The GPU is busy only during the short
minibatch-update phase and idles through the (dominant) rollout phase. Classic
control has no GPU at all (tiny MLP on CPU). So per-run, the GPU is mostly idle.

## What we do about it
**Pack many runs onto one GPU.** Each run needs ~1 GB and ~15% SM, so ~6 fit and
overlap (one's rollout hides another's update), pushing a single GPU toward full
use instead of holding 6 GPUs at 15% each. Implemented:
- `experiments/run_packed.py` — launches N `atari_ac` runs concurrently on the one
  GPU SLURM assigned (shared CUDA context), bounded so env subprocesses don't
  oversubscribe cores.
- `slurm/atari_packed.slurm` — 1 GPU + 72 cores, runs 6 ablation runs together.
This replaced 6 wasteful 1-GPU jobs (9586029–34) with one GPU-efficient job
(9586044), freeing 5 GPUs for other work on the shared cluster.

## Other levers (situational)
- **`--num_envs` ↑** (more parallel env stepping → bigger forward batch + faster
  rollout) — bounded by CPU cores; the main knob for single-run throughput.
- **larger `--minibatch` / `--update_epochs`** — more GPU work in the update phase
  (raises average util), but the rollout phase still idles.
- **EnvPool** (C++ batched Atari) or GPU-side envs — removes the env bottleneck so
  the GPU becomes the limiter; needs a dependency install. Biggest single-run win
  if we want one run to saturate a GPU.
- **CUDA MPS** — can improve fairness when packing many processes on one GPU.

## Guidance
- Classic control: CPU only, no GPU (don't request one).
- Atari: don't give a whole GPU to one small actor-only run — **pack** (via
  `run_packed.py`) or bump `num_envs`/batch. Reserve dedicated full GPUs for the
  compute-bound PPO-critic training, where utilization is higher.
