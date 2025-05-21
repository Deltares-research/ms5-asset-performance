#!/bin/bash

source .venv_torch/bin/activate


lr_exps=(-4)

epochs=(100_000)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/torch_results"
mkdir -p "$LOG_DIR"
#rm -f "$LOG_DIR"/*

log_file="$LOG_DIR/log.txt"

summary_file="$LOG_DIR/summary.txt"

for lr_exp in "${lr_exps[@]}"; do
  for ep in "${epochs[@]}"; do
    lr=$(echo "10 ^ $lr_exp" | bc -l)
    echo "Running: lr_exp=$lr_exp and epochs=$ep" | tee -a "$log_file"
    result=$(python main/case_study_2025/torch_train.py --lr "$lr" --epochs "$ep" 2>&1 | tee /dev/tty)
    result="${result}\n"
    echo "$result" >> "$log_file"
    summary=$(echo "$result" | grep "\[SUMMARY\]")
    echo "$summary" >> "$summary_file"
    echo "---------------------------------------------" >> "$log_file"
  done
done

echo "Grid search complete."
