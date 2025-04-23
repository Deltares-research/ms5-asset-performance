#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
#cd $SCRIPT_DIR

# Define Python path (Windows venv)
PYTHON="$(cd "$SCRIPT_DIR/../../.." && pwd)/.venv/Scripts/python.exe"
#echo "Using Python at: $PYTHON"

JSON_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)/examples/reliability/bash"
mkdir -p "$JSON_DIR"
rm -f "$JSON_DIR"/*

# Create log folder and clear past logs
LOG_DIR="$SCRIPT_DIR/bash_logs"
mkdir -p "$LOG_DIR"
#rm -f "$LOG_DIR"/*

 Define input values
points=()
for x in $(seq -3.0 0.3 3.0); do
    points+=("$x")
done

# Run loop
for point in "${points[@]}"; do
    echo "Running for water level point = $point"
    POINT=$point "$PYTHON" -m src.reliability_models.dsheetpiling.fragility_point >> "$LOG_DIR/run_${point}.log" 2>&1
done

echo "✅    Run finished."
