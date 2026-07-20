"""Pack several atari_ac runs onto ONE GPU (they each use ~15% SM / ~1GB, so
many fit) — turns wasteful 1-GPU-per-tiny-job into full-GPU utilization.
All child processes share the single GPU SLURM gave us (CUDA_VISIBLE_DEVICES);
concurrency is bounded so we don't oversubscribe CPU cores for env stepping.
Usage: python experiments/run_packed.py <specs.json> [max_parallel]
Each spec: {env_id, actor_loss, d_kind, critic_ckpt, seed, num_envs, total_timesteps, tag}
"""
import os, sys, json, time, subprocess
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "src", "atari_ac.py")


def run(spec, idx, stagger=4):
    time.sleep(idx * stagger)
    args = [PY, SCRIPT]
    for k, v in spec.items():
        args += [f"--{k}", str(v)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1")
    ld = os.path.join(REPO, "results", "logs"); os.makedirs(ld, exist_ok=True)
    with open(os.path.join(ld, spec["tag"] + ".log"), "w") as lf:
        rc = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
    return spec["tag"], rc


def main():
    specs = json.load(open(sys.argv[1]))
    mp = int(sys.argv[2]) if len(sys.argv) > 2 else len(specs)
    print(f"[packed] {len(specs)} runs sharing 1 GPU, max_parallel={mp}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=mp) as ex:
        for f in [ex.submit(run, s, i) for i, s in enumerate(specs)]:
            tag, rc = f.result(); print(f"[done] {tag} rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[packed] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
