#!/bin/bash
set -e

echo "*****Running text trainer"

# Define your arguments here
TASK_ID="b62e45ff-8cc7-4a01-858d-657bdd82a936"
MODEL="unsloth/Phi-3-mini-4k-instruct"
DATASET="https://gradients.s3.eu-north-1.amazonaws.com/2a79586c120c3ade_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20251013%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251013T100948Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=d577fe843df09136e271fd50137ce2d2b046b6781ab5681a34f5449e39ea7de2"
DATASET_TYPE={
    "field_system": null,
    "field_instruction": "instruct",
    "field_input": null,
    "field_output": "output",
    "format": null,
    "no_input_format": null,
    "system_format": null 
}
TASK_TYPE="InstructTextTask"  # or DpoTask, GrpoTask
HOURS_TO_COMPLETE=2
FILE_FORMAT="s3"  # csv, json, hf, s3
EXPECTED_REPO_NAME="imsotired"
MAX_DATA_SIZE=-1
MAX_STEPS=-1
RETRIES=5
MIN_STEPS=100
REG_RATIO=1.0

# Create training request file
echo "Creating training request file..."
python -m create_train_request_file \
    --dataset-type "$DATASET_TYPE" \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --min-steps "$MIN_STEPS" \
    --max-steps "$MAX_STEPS" \
    --max-data-size "$MAX_DATA_SIZE"
echo "Training request file created."

# download dataset

echo "Downloading dataset..."
python -m trainer_downloader \
    --task-id "$TASK_ID" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --file-format "$FILE_FORMAT" \
    --task-type "$TASK_TYPE" \
echo "Dataset downloaded."

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