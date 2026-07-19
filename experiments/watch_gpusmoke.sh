#!/bin/bash
J=9479067; cd /u/ad11/rl_env
for i in $(seq 1 120); do
  st=$(squeue -j $J -h -o "%T" 2>/dev/null)
  o=slurm/rrebel_gpu_smoke_${J}.out
  [ -e "$o" ] && grep -q "SUMMARY\|it=0005\|Traceback" "$o" 2>/dev/null && break
  [ -z "$st" ] && break
  sleep 30
done
echo "=== GPU smoke status $(date) ==="
o=slurm/rrebel_gpu_smoke_${J}.out
[ -e "$o" ] && { grep -E "host=|it=|>>>|SUMMARY|Error|Traceback" "$o" | head -20; } || echo "no output"
