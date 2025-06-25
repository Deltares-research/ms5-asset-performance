#!/bin/bash

# Load environment variables if .env file exists
if [ -f "$(dirname "$0")/.env" ]; then
    source "$(dirname "$0")/.env"
fi

# Check if conda environment is specified, otherwise use virtual environment
if [ -n "$CONDA_ENV_NAME" ]; then
    # Initialize conda for bash (required for conda activate to work in scripts)
    eval "$(conda shell.bash hook)"
    # Activate the conda environment
    CONDA_ENV_NAME=${CONDA_ENV_NAME:-"Subsoil"}
    conda activate "$CONDA_ENV_NAME"
else
    # Fall back to virtual environment
    source .venv_torch/bin/activate
fi

lr_exps=(-2)
epochs=(1_00)
ranks=(1 2)

# Get the project root directory (2 levels up from script location)
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Change working directory to project root
cd "$PROJECT_ROOT"
# Add project root to PYTHONPATH so Python can find the src module
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

LOG_DIR="main/case_study_2025/results_moments/srg_moments_20250617_101504/gpr"
mkdir -p "$LOG_DIR"
#rm -f "$LOG_DIR"/*

log_file="$LOG_DIR/log.txt"

summary_file="$LOG_DIR/summary.txt"

for lr_exp in "${lr_exps[@]}"; do
  for ep in "${epochs[@]}"; do
    for rank in "${ranks[@]}"; do
      lr=$(echo "10 ^ $lr_exp" | bc -l)
      echo "Running: lr_exp=$lr_exp and epochs=$ep and rank=$rank" | tee -a "$log_file"
      result=$(python "main/case_study_2025/train/srg/gpr_train.py" --lr "$lr" --epochs "$ep" --rank "$rank" 2>&1 | tee /dev/tty)
      result="${result}\n"
      echo "$result" >> "$log_file"
      summary=$(echo "$result" | grep "\[SUMMARY\]")
      echo "$summary" >> "$summary_file"
      echo "---------------------------------------------" >> "$log_file"
    done
  done
done

echo "Script finished."