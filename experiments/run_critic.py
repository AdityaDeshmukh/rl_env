"""Parallel launcher for src/critic_ablation.py.
Usage: python experiments/run_critic.py <specs.json> [max_parallel]"""
import sys, json
from _launcher import launch

if __name__ == "__main__":
    jobs = json.load(open(sys.argv[1]))
    launch("src/critic_ablation.py", jobs, int(sys.argv[2]) if len(sys.argv) > 2 else 8)
