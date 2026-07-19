#!/bin/bash
JOB=9452472
cd /u/ad11/rl_env
# poll until the job leaves the queue (finished/timed out), hard cap ~3.75h
for i in $(seq 1 45); do
  squeue -j $JOB -h -o "%T" 2>/dev/null | grep -q . || break
  sleep 300
done
echo "=== job $JOB no longer running; aggregating $(date) ==="
python experiments/atari_table.py results/atari atari_results.tex 2>&1
