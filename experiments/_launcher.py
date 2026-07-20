"""Shared subprocess fan-out for the experiment runners (run_matrix / run_critic /
run_packed). Runs each job dict as `python <script> --k v ...`, with staggered
starts (dodges NFS import races), capped concurrency, and per-tag logging."""
import os, sys, time, subprocess
from concurrent.futures import ThreadPoolExecutor

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ENV = dict(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def launch(script, jobs, max_parallel=8, stagger=1.5):
    """script: repo-relative path (e.g. 'src/train.py'). jobs: list of arg-dicts,
    each with a 'tag'. Returns list of (tag, returncode)."""
    script = os.path.join(REPO, script)
    env = dict(os.environ, **_ENV)
    logdir = os.path.join(REPO, "results", "logs")
    os.makedirs(logdir, exist_ok=True)

    def run(job, idx):
        time.sleep(idx * stagger)
        args = [sys.executable, script]
        for k, v in job.items():
            args += [f"--{k}", str(v)]
        with open(os.path.join(logdir, job["tag"] + ".log"), "w") as lf:
            rc = subprocess.run(args, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=REPO).returncode
        return job["tag"], rc

    print(f"[launch] {os.path.basename(script)} x{len(jobs)} max_parallel={max_parallel}", flush=True)
    t0 = time.time(); results = []
    with ThreadPoolExecutor(max_workers=max_parallel) as ex:
        for f in [ex.submit(run, j, i % max_parallel) for i, j in enumerate(jobs)]:
            tag, rc = f.result(); results.append((tag, rc))
            print(f"[done] {tag} rc={rc} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[launch] all done in {time.time()-t0:.0f}s", flush=True)
    return results
