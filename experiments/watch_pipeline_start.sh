#!/bin/bash
J=9481426; cd /u/ad11/rl_env
for i in $(seq 1 240); do
  st=$(squeue -j $J -h -o "%T" 2>/dev/null)
  o=$(ls slurm/rrebel_critic_atari_${J}.out 2>/dev/null)
  if [ -n "$o" ]; then
    grep -qE "it=0006|SUMMARY|Traceback|Error|CUDA" "$o" 2>/dev/null && break
  fi
  [ -z "$st" ] && break
  sleep 45
done
echo "=== pipeline start check $(date) ==="
o=slurm/rrebel_critic_atari_${J}.out
[ -e "$o" ] && { echo "--- head ---"; head -2 "$o"; echo "--- progress ---"; grep -E "host=|>>>|it=|SUMMARY|Traceback|Error|CUDA" "$o" | head -14; } || echo "job never started / no output (state: $(squeue -j $J -h -o '%T' 2>/dev/null || echo gone))"
