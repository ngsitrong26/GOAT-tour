import os
import json
import argparse

import training_paths as train_paths

def create_training_request_file(output_path: str = "training_request.json"):
    """
    Create a training request JSON file for the text trainer.
    """
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--min-steps", type=int, default=100, help="Minimum training steps")
    parser.add_argument("--max-steps", type=int, default=1000, help="Maximum training steps")
    parser.add_argument("--max-data-size", type=int, default=-1, help="Maximum data size to use")
    args = parser.parse_args()

    model_path = str(train_paths.get_text_base_model_path(args.model))
    dataset_path = train_paths.get_text_dataset_path(args.task_id)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    output_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")

    training_request = {
        "train_request": {
            "task_id": args.task_id,
            "dataset": dataset_path,
            "model_path": model_path,
            "model_name": args.model,
            "dataset_type": args.dataset_type,
            "max_data_size": args.max_data_size,
            "max_length": 2048,
            "min_steps": args.min_steps,
            "max_steps": args.max_steps
        }
    }
    
    # Write to file
    with open(output_path, "w") as file:
        json.dump(training_request, indent=2, fp=file)
    
    print(f"Training request file created at: {output_path}")
    return output_path