"""Parallel launcher for src/train.py over a JSON matrix.
Usage: python experiments/run_matrix.py <matrix.json> [max_parallel]
Each config is a dict of --flag values for src/train.py; it MUST include "tag"."""
import sys, json
from _launcher import launch

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    launch("src/train.py", jobs, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
