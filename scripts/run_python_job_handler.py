import json
import os
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
import toml
from dotenv import load_dotenv
from fiber.logging_utils import get_logger
from huggingface_hub import HfApi, login as hf_login

from core import constants as cst
from core.config.config_handler import create_dataset_entry, save_config, save_config_toml
from core.config.config_handler import update_flash_attention, update_model_info
from core.dataset.prepare_diffusion_dataset import prepare_dataset
from core.models.utility_models import (
    DiffusionJob, TextJob, 
    TextDatasetType, DpoDatasetType, GrpoDatasetType, InstructTextDatasetType, ChatTemplateDatasetType,
    ImageModelType,
    FileFormat
)

logger = get_logger(__name__)

load_dotenv()

@dataclass
class TrainingEnvironment:
    """Environment setup for training without Docker"""
    huggingface_token: str
    wandb_token: str
    job_id: str
    
    def setup(self):
        """Authenticate with HuggingFace and W&B"""
        if self.huggingface_token:
            hf_login(token=self.huggingface_token)
            print("Logged into Hugging Face")
        
        if self.wandb_token:
            os.environ['WANDB_API_KEY'] = self.wandb_token
            print("W&B token set")
            
def _dpo_format_prompt(row, format_str):
    result = format_str
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_chosen(row, format_str):
    result = format_str
    if "{chosen}" in format_str and cst.DPO_DEFAULT_FIELD_CHOSEN in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_CHOSEN]):
        result = result.replace("{chosen}", str(row[cst.DPO_DEFAULT_FIELD_CHOSEN]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result


def _dpo_format_rejected(row, format_str):
    result = format_str
    if "{rejected}" in format_str and cst.DPO_DEFAULT_FIELD_REJECTED in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_REJECTED]):
        result = result.replace("{rejected}", str(row[cst.DPO_DEFAULT_FIELD_REJECTED]))
    if "{prompt}" in format_str and cst.DPO_DEFAULT_FIELD_PROMPT in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_PROMPT]):
        result = result.replace("{prompt}", str(row[cst.DPO_DEFAULT_FIELD_PROMPT]))
    if "{system}" in format_str and cst.DPO_DEFAULT_FIELD_SYSTEM in row and pd.notna(row[cst.DPO_DEFAULT_FIELD_SYSTEM]):
        result = result.replace("{system}", str(row[cst.DPO_DEFAULT_FIELD_SYSTEM]))
    return result

def _adapt_columns_for_grpo_dataset(dataset_path: str, dataset_type: GrpoDatasetType):
    """
    Transform a GRPO JSON dataset file to match axolotl's `prompt` expected column name.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: GrpoDatasetType with field mappings
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    df = df.rename(columns={dataset_type.field_prompt: cst.GRPO_DEFAULT_FIELD_PROMPT})
    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)
            
def _adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, apply_formatting: bool = False):
    """
    Transform a DPO JSON dataset file to match axolotl's `chatml.argilla` expected column names.

    Args:
        dataset_path: Path to the JSON dataset file
        dataset_type: DpoDatasetType with field mappings
        apply_formatting: If True, apply formatting templates to the content
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    column_mapping = {
        dataset_type.field_prompt: cst.DPO_DEFAULT_FIELD_PROMPT,
        dataset_type.field_system: cst.DPO_DEFAULT_FIELD_SYSTEM,
        dataset_type.field_chosen: cst.DPO_DEFAULT_FIELD_CHOSEN,
        dataset_type.field_rejected: cst.DPO_DEFAULT_FIELD_REJECTED
    }
    df = df.rename(columns=column_mapping)

    if apply_formatting:
        if dataset_type.prompt_format and dataset_type.prompt_format != "{prompt}":
            format_str = dataset_type.prompt_format
            df[cst.DPO_DEFAULT_FIELD_PROMPT] = df.apply(lambda row: _dpo_format_prompt(row, format_str), axis=1)
        if dataset_type.chosen_format and dataset_type.chosen_format != "{chosen}":
            format_str = dataset_type.chosen_format
            df[cst.DPO_DEFAULT_FIELD_CHOSEN] = df.apply(lambda row: _dpo_format_chosen(row, format_str), axis=1)
        if dataset_type.rejected_format and dataset_type.rejected_format != "{rejected}":
            format_str = dataset_type.rejected_format
            df[cst.DPO_DEFAULT_FIELD_REJECTED] = df.apply(lambda row: _dpo_format_rejected(row, format_str), axis=1)

    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)

def prepare_training_environment(job: TextJob, config_path: str):
    """Prepare environment and dataset for training"""
    # Set up authentication
    env = TrainingEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id
    )
    env.setup()
    
    # Adapt dataset columns if needed
    if job.file_format == FileFormat.JSON:
        if isinstance(job.dataset_type, DpoDatasetType):
            _adapt_columns_for_dpo_dataset(job.dataset, job.dataset_type, True)
        elif isinstance(job.dataset_type, GrpoDatasetType):
            _adapt_columns_for_grpo_dataset(job.dataset, job.dataset_type)
    
    # Copy dataset to working directory if needed
    if job.file_format != FileFormat.HF:
        dataset_filename = os.path.basename(job.dataset)
        dest_path = os.path.join(os.getcwd(), dataset_filename)
        if not os.path.exists(dest_path):
            import shutil
            shutil.copy(job.dataset, dest_path)
            print(f"Copied dataset to {dest_path}")
            
def create_reward_funcs_file(reward_funcs: list[str], task_id: str, destination_dir: str = cst.CONFIG_DIR) -> list[str]:
    """
    Create a Python file with reward functions for GRPO training.

    Args:
        reward_funcs: List of strings containing Python reward function implementations
        task_id: Unique task identifier
    """
    filename = f"rewards_{task_id}"
    filepath = os.path.join(destination_dir, f"{filename}.py")

    func_names = []
    for reward_func in reward_funcs:
        if "def " in reward_func:
            func_name = reward_func.split("def ")[1].split("(")[0].strip()
            func_names.append(func_name)

    with open(filepath, "w") as f:
        f.write("# Auto-generated reward functions file\n\n")
        for reward_func in reward_funcs:
            f.write(f"{reward_func}\n\n")

    return filename, func_names
            
def _load_and_modify_config(
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    task_id: str,
    expected_repo_name: str | None,
) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    if isinstance(dataset_type, InstructTextDatasetType | DpoDatasetType | ChatTemplateDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH
    elif isinstance(dataset_type, GrpoDatasetType):
        config_path = cst.CONFIG_TEMPLATE_PATH_GRPO

    logger.info("Loading config template")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config["datasets"] = []

    dataset_entry = create_dataset_entry(dataset, dataset_type, file_format)
    config["datasets"].append(dataset_entry)

    if isinstance(dataset_type, DpoDatasetType):
        config["rl"] = "dpo"
    elif isinstance(dataset_type, GrpoDatasetType):
        filename, reward_funcs_names = create_reward_funcs_file(
            [reward_function.reward_func for reward_function in dataset_type.reward_functions], task_id
            )
        config["trl"]["reward_funcs"] = [f"{filename}.{func_name}" for func_name in reward_funcs_names]
        config["trl"]["reward_weights"] = [reward_function.reward_weight for reward_function in dataset_type.reward_functions]

    config = update_flash_attention(config, model)
    config = update_model_info(config, model, task_id, expected_repo_name)
    config["mlflow_experiment_name"] = dataset

    return config


def run_text_training(job: TextJob):
    """Run text model training directly in Python"""
    print("=" * 80)
    print("STARTING TEXT MODEL TRAINING")
    print("=" * 80)
    
    # Create config
    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.yml")
    config = _load_and_modify_config(
        job.dataset,
        job.model,
        job.dataset_type,
        job.file_format,
        job.job_id,
        job.expected_repo_name,
    )
    save_config(config, config_path)
    print(f"Config saved to {config_path}")
    
    # Prepare environment
    prepare_training_environment(job, config_path)
    
    # Create reward functions file for GRPO if needed
    if isinstance(job.dataset_type, GrpoDatasetType):
        reward_file = f"rewards_{job.job_id}.py"
        reward_path = os.path.join(cst.CONFIG_DIR, reward_file)
        # Copy to axolotl src directory if it exists
        axolotl_src = Path("axolotl/src")
        if axolotl_src.exists():
            import shutil
            shutil.copy(reward_path, axolotl_src / reward_file)
    
    try:
        # Run training using accelerate + axolotl
        cmd = [
            "accelerate", "launch",
            "-m", "axolotl.cli.train",
            config_path
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        # Run training process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream logs
        for line in process.stdout:
            print(line, end='')
        
        # Wait for completion
        return_code = process.wait()
        
        if return_code != 0:
            raise RuntimeError(f"Training failed with exit code {return_code}")
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Make repository public
        repo = config.get("hub_model_id")
        if repo:
            hf_api = HfApi(token=cst.HUGGINGFACE_TOKEN)
            hf_api.update_repo_visibility(
                repo_id=repo,
                private=False,
                token=cst.HUGGINGFACE_TOKEN
            )
            print(f"Repository {repo} is now public")

def _load_and_modify_config_diffusion(job: DiffusionJob) -> dict:
    """
    Loads the config template and modifies it to create a new job config.
    """
    logger.info("Loading config template")
    if job.model_type == ImageModelType.SDXL:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_SDXL, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = job.model
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    elif job.model_type == ImageModelType.FLUX:
        with open(cst.CONFIG_TEMPLATE_PATH_DIFFUSION_FLUX, "r") as file:
            config = toml.load(file)
        config["pretrained_model_name_or_path"] = f"{cst.CONTAINER_FLUX_PATH}/flux_unet_{job.model.replace('/', '_')}.safetensors"
        config["train_data_dir"] = f"/dataset/images/{job.job_id}/img/"
        config["huggingface_token"] = cst.HUGGINGFACE_TOKEN
        config["huggingface_repo_id"] = f"{cst.HUGGINGFACE_USERNAME}/{job.expected_repo_name or str(uuid.uuid4())}"
    else:
        logger.error(f"Unknown model type: {job.model_type}")
    return config

def run_diffusion_training(job: DiffusionJob):
    """Run diffusion model training directly in Python"""
    print("=" * 80)
    print("STARTING DIFFUSION MODEL TRAINING")
    print("=" * 80)
    
    # Create config
    config_path = os.path.join(cst.CONFIG_DIR, f"{job.job_id}.toml")
    config = _load_and_modify_config_diffusion(job)
    save_config_toml(config, config_path)
    print(f"Config saved to {config_path}")
    
    # Set up authentication
    env = TrainingEnvironment(
        huggingface_token=cst.HUGGINGFACE_TOKEN,
        wandb_token=cst.WANDB_TOKEN,
        job_id=job.job_id
    )
    env.setup()
    
    # Prepare dataset
    prepare_dataset(
        training_images_zip_path=job.dataset_zip,
        training_images_repeat=(
            cst.DIFFUSION_SDXL_REPEATS if job.model_type == ImageModelType.SDXL
            else cst.DIFFUSION_FLUX_REPEATS
        ),
        instance_prompt=cst.DIFFUSION_DEFAULT_INSTANCE_PROMPT,
        class_prompt=cst.DIFFUSION_DEFAULT_CLASS_PROMPT,
        job_id=job.job_id,
    )
    
    # Download FLUX unet if needed
    if job.model_type == ImageModelType.FLUX:
        from miner.utils import download_flux_unet
        download_flux_unet(job.model)
    
    try:
        # Run training script
        cmd = ["python", "train_diffusion.py", "--config", config_path]
        
        print(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Stream logs
        for line in process.stdout:
            print(line, end='')
        
        return_code = process.wait()
        
        if return_code != 0:
            raise RuntimeError(f"Training failed with exit code {return_code}")
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise
    
    finally:
        # Cleanup
        import shutil
        train_data_path = f"{cst.DIFFUSION_DATASET_DIR}/{job.job_id}"
        if os.path.exists(train_data_path):
            shutil.rmtree(train_data_path)
            print(f"Cleaned up training data at {train_data_path}")

def create_job_text(
    job_id: str,
    dataset: str,
    model: str,
    dataset_type: TextDatasetType,
    file_format: FileFormat,
    expected_repo_name: str | None,
):
    return TextJob(
        job_id=job_id,
        dataset=dataset,
        model=model,
        dataset_type=dataset_type,
        file_format=file_format,
        expected_repo_name=expected_repo_name,
    )

# Example usage
if __name__ == "__main__":
    # For text training
    
    job = TextJob.model_validate(
        {
            "model": "unsloth/Phi-3-mini-4k-instruct",
            "job_id": "b62e45ff-8cc7-4a01-858d-657bdd82a936",
            "expected_repo_name": "imsotired",
            "dataset": "https://gradients.s3.eu-north-1.amazonaws.com/2a79586c120c3ade_train_data.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAVVZOOA7SA4UOFLPI%2F20251013%2Feu-north-1%2Fs3%2Faws4_request&X-Amz-Date=20251013T100948Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=d577fe843df09136e271fd50137ce2d2b046b6781ab5681a34f5449e39ea7de2",
            "dataset_type": {
                "system_prompt": "",
                "system_format": "{system}",
                "field_system": None,
                "field_instruction": "instruct",
                "field_input": None,
                "field_output": "output",
                "format": None,
                "no_input_format": None,
                "field": None
            },
            "file_format": "s3",
        }
    )
    # job = create_job_text(
    #     job_id=str(uuid.uuid4()),
    #     dataset="/path/to/dataset.json",
    #     model="meta-llama/Llama-2-7b-hf",
    #     dataset_type=your_dataset_type,
    #     file_format=FileFormat.JSON,
    #     expected_repo_name="my-finetuned-model"
    # )
    run_text_training(job)
    
    # For diffusion training
    # diffusion_job = create_job_diffusion(...)
    # run_diffusion_training(diffusion_job)