#!/bin/bash
cd /u/ad11/rl_env
for i in $(seq 1 60); do
  b=$(grep -q "all done" results/lunar_master.log 2>/dev/null && echo 1 || echo 0)
  c=$(grep -q "all done" results/critic_master.log 2>/dev/null && echo 1 || echo 0)
  [ "$b" = "1" ] && [ "$c" = "1" ] && break
  sleep 60
done
echo "=== (b)/(c) done $(date) ==="
echo "lunar: $(ls results/lunar/*.json 2>/dev/null|wc -l)/18  critic: $(ls results/critic/*.json 2>/dev/null|wc -l)/24"
