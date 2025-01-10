#!/bin/bash

set -e

MODEL_DIR="models"  
SEED=42  

for MODEL_FILE in "$MODEL_DIR"/*.pt; do
    if [ -f "$MODEL_FILE" ]; then
        echo "Running model baseline: $MODEL_FILE"
        python baseline.py --seed $SEED --model_name "$MODEL_FILE" --maneuver "$MANEUVER"
    else
        echo "No .pt files found in $MODEL_DIR"
    fi
done