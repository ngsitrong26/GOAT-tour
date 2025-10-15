#!/bin/bash
set -e

echo "*****Running text trainer"

# Define your arguments here
TASK_ID="your-task-id"
MODEL="your-model-name"
DATASET="your-dataset-path"
DATASET_TYPE='{"type": "your-type"}'  # JSON string
TASK_TYPE="InstructTextTask"  # or DpoTask, GrpoTask
HOURS_TO_COMPLETE=2.5
FILE_FORMAT="s3"  # csv, json, hf, s3
EXPECTED_REPO_NAME="your-repo-name"
MAX_DATA_SIZE=1000
MAX_STEPS=500
RETRIES=5
MIN_STEPS=100
REG_RATIO=1.01

# Run the text trainer with all arguments
python -m text_trainer \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --dataset-type "$DATASET_TYPE" \
    --task-type "$TASK_TYPE" \
    --hours-to-complete "$HOURS_TO_COMPLETE" \
    --file-format "$FILE_FORMAT" \
    --expected-repo-name "$EXPECTED_REPO_NAME" \
    --max-data-size "$MAX_DATA_SIZE" \
    --max-steps "$MAX_STEPS" \
    --retries "$RETRIES" \
    --min-steps "$MIN_STEPS" \
    --reg-ratio "$REG_RATIO"