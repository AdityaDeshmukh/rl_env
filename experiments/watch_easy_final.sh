#!/bin/bash
JOB=9455024
cd /u/ad11/rl_env
for i in $(seq 1 48); do
  squeue -j $JOB -h -o "%T" 2>/dev/null | grep -q . || break
  sleep 300
done
echo "=== job $JOB done $(date) ==="
for f in results/atari_easy/*.csv; do
  echo "-- $(basename $f .csv) (last) --"; tail -1 "$f"
done
python experiments/atari_table.py results/atari_easy atari_easy_results.tex 2>&1 | tail -12
