#!/bin/bash

#source .venv_torch/bin/activate
source /home/amavrits/.virtualenvs/ms5-asset-performance/bin/activate

estimators=(1000)
maxdepths=(10)
lrs=(0.05)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/results/srg/xgb"
mkdir -p "$LOG_DIR"

log_file="$LOG_DIR/log.txt"

summary_file="$LOG_DIR/summary.txt"

for est in "${estimators[@]}"; do
  for d in "${maxdepths[@]}"; do
    for lr in "${lrs[@]}"; do
      echo "Running: n_estimators=$est | max_depth=$d | lr=$lr" | tee -a "$log_file"
      result=$(python "$SCRIPT_DIR/train/srg/xgb_train.py" --n-estimators "$est" --max-depth "$d" --lr "$lr" 2>&1 | tee /dev/tty)
      result="${result}\n"
      echo "$result" >> "$log_file"
      summary=$(echo "$result" | grep "\[SUMMARY\]")
      echo "$summary" >> "$summary_file"
      echo "---------------------------------------------" >> "$log_file"
    done
  done
done

echo "Grid search complete."
