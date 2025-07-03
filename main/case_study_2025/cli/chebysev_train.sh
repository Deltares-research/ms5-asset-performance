#!/bin/bash

source .venv_torch/bin/activate
#source /home/amavrits/.virtualenvs/ms5-asset-performance/bin/activate

lr_exps=(-5)
epochs=(100_000)
fullprofile=true

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/results/srg/torch"
mkdir -p "$LOG_DIR"
#rm -f "$LOG_DIR"/*

log_file="$LOG_DIR/log.txt"

summary_file="$LOG_DIR/summary.txt"

for lr_exp in "${lr_exps[@]}"; do
  for ep in "${epochs[@]}"; do
    lr=$(echo "10 ^ $lr_exp" | bc -l)
    echo "Running: lr_exp=$lr_exp, epochs=$ep and fullprofile=$fullprofile" | tee -a "$log_file"
    if [ "$fullprofile" = "true" ]; then
      result=$(python -m main.case_study_2025.train.srg.chebysev_train --lr "$lr" --epochs "$ep" --full-profile 2>&1 | tee /dev/tty)
    else
      result=$(python -m main.case_study_2025.train.srg.chebysev_train --lr "$lr" --epochs "$ep" --no-full-profile 2>&1 | tee /dev/tty)
    fi
    result="${result}\n"
    echo "$result" >> "$log_file"
    summary=$(echo "$result" | grep "\[SUMMARY\]")
    echo "$summary" >> "$summary_file"
    echo "---------------------------------------------" >> "$log_file"
  done
done

echo "Grid search complete."
