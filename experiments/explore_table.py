"""Summarize the exploration bake-off (results/explore) into a per-env, per-mode
table: final and best eval (mean±std over seeds), plus solve counts."""
import json, glob, math, sys, os
from collections import defaultdict

RES = sys.argv[1] if len(sys.argv) > 1 else "results/explore"
SOLVE = {"Acrobot-v1": -110.0, "Pendulum-v1": -300.0, "LunarLander-v3": 200.0}
MODES = ["onpolicy", "temp", "eps", "topm", "mixed", "elite", "adaptive"]


def ms(xs):
    m = sum(xs) / len(xs)
    return m, (sum((x - m) ** 2 for x in xs) / max(1, len(xs) - 1)) ** 0.5


def main():
    rows = defaultdict(list)   # (env, variant) -> [(final, best)]
    for f in glob.glob(os.path.join(RES, "*.json")):
        d = json.load(open(f))
        env, variant, _ = d["name"].split("__")
        rows[(env, variant)].append((d["final_mean"], d["best_eval"]))
    for env in ["Acrobot-v1", "Pendulum-v1", "LunarLander-v3"]:
        thr = SOLVE[env]
        print(f"\n===== {env} (solve >= {thr}) =====")
        variants = ([f"hub_{m}" for m in MODES] + ["grpo_ref"]
                    + ([f"l1_{m}" for m in ["onpolicy", "elite", "adaptive"]] if env == "Acrobot-v1" else []))
        for v in variants:
            xs = rows.get((env, v))
            if not xs:
                continue
            fm, fs = ms([a for a, _ in xs]); bm, bs = ms([b for _, b in xs])
            solved = sum(1 for a, _ in xs if a >= thr)
            print(f"  {v:14s} final={fm:8.1f}+/-{fs:6.1f}  best={bm:8.1f}+/-{bs:6.1f}  solved={solved}/{len(xs)}")


if __name__ == "__main__":
    main()
