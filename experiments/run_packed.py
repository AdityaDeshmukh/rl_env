"""Pack several atari_ac runs onto ONE GPU (each uses ~15% SM / ~1GB, so many
fit) — turns wasteful 1-GPU-per-tiny-job into full-GPU utilization. All children
share the single GPU SLURM assigned (CUDA_VISIBLE_DEVICES); concurrency is
bounded so we don't oversubscribe CPU cores for env stepping. See GPU_EFFICIENCY.md.
Usage: python experiments/run_packed.py <specs.json> [max_parallel]"""
import sys, json
from _launcher import launch

if __name__ == "__main__":
    specs = json.load(open(sys.argv[1]))
    launch("src/atari_ac.py", specs, int(sys.argv[2]) if len(sys.argv) > 2 else len(specs), stagger=4)
