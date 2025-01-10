#!/bin/bash

set -e

MODEL_DIR="../experiment/train_for_evaluate/models"  
SEED=42  

for MODEL_FILE in "$MODEL_DIR"/*.pt; do
    if [ -f "$MODEL_FILE" ]; then
        echo "Processing model: $MODEL_FILE"
        python render.py --seed $SEED --model_name "$MODEL_FILE"
    else
        echo "No .pt files found in $MODEL_DIR"
    fi
done