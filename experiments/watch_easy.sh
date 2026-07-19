#!/bin/bash
JOB=9455024
cd /u/ad11/rl_env
# 1) wait for it to start (or vanish)
for i in $(seq 1 90); do
  st=$(squeue -j $JOB -h -o "%T" 2>/dev/null)
  [ -z "$st" ] && { echo "job gone before producing data"; break; }
  [ "$st" = "RUNNING" ] && break
  sleep 60
done
echo "=== running; watching learning trend $(date) ==="
# 2) once running, exit when we have a clear trend (>=700k steps somewhere) or job ends
for i in $(seq 1 60); do
  gone=$(squeue -j $JOB -h -o "%T" 2>/dev/null)
  maxstep=$(grep -h 'step=' results/logs/Pong__*.log results/logs/Boxing__*.log 2>/dev/null \
            | grep -oE 'step= *[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
  maxstep=${maxstep:-0}
  [ -z "$gone" ] && { echo "job finished"; break; }
  [ "$maxstep" -ge 700000 ] && { echo "reached $maxstep steps -> trend available"; break; }
  sleep 180
done
echo "=== TREND $(date) ==="
for f in results/logs/Pong__*.log results/logs/Boxing__*.log; do
  [ -e "$f" ] && { echo "-- $(basename $f .log) --"; grep 'upd=' "$f" | awk 'NR%8==1' | tail -6; }
done
python experiments/atari_table.py results/atari_easy atari_easy_results.tex 2>&1 | tail -12
