"""Robust parallel launcher for src/train.py.
Runs a JSON list of configs as subprocesses with capped concurrency + staggered
starts (avoids NFS import races on the cluster). Collects JSON summaries.

Usage:
  python experiments/run_matrix.py experiments/matrices/main.json [max_parallel]
Each config is a dict of --flag values for src/train.py; it MUST include "tag".
"""
import os, sys, json, time, subprocess
from concurrent.futures import ThreadPoolExecutor

PY = sys.executable
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(REPO, "src", "train.py")


def run_one(job, idx, stagger=1.5):
    time.sleep(idx * stagger)
    args = [PY, TRAIN]
    for k, v in job.items():
        args += [f"--{k}", str(v)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    logdir = os.path.join(REPO, "results", "logs")
    os.makedirs(logdir, exist_ok=True)
    with open(os.path.join(logdir, job["tag"] + ".log"), "w") as lf:
        p = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO)
    return job["tag"], p.returncode


def main():
    matrix = sys.argv[1]
    max_par = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    jobs = json.load(open(matrix))
    print(f"[matrix] {matrix}: {len(jobs)} jobs, max_parallel={max_par}", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max_par) as ex:
        futs = [ex.submit(run_one, j, i % max_par) for i, j in enumerate(jobs)]
        for f in futs:
            tag, rc = f.result()
            print(f"[done] {tag} rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[matrix] all done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
