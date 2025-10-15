import json
import os
import requests
import pandas as pd

import core.constants as cst
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import GrpoDatasetType

from urllib.parse import urlparse


def download_s3_file(file_url: str, save_path: str = None, tmp_dir: str = "/tmp", download: bool = True) -> str:
    """Download a file from an S3 URL and save it locally, or just return the expected path.

    Args:
        file_url (str): The URL of the file to download.
        save_path (str, optional): The path where the file should be saved. If a directory is provided,
            the file will be saved with its original name in that directory. If a file path is provided,
            the file will be saved at that exact location. Defaults to None.
        tmp_dir (str, optional): The temporary directory to use when save_path is not provided.
            Defaults to "/tmp".
        download (bool, optional): Whether to actually download the file. If False, only returns
            the expected file path. Defaults to True.

    Returns:
        str: The local file path where the file was saved (or would be saved).

    Raises:
        Exception: If the download fails with a non-200 status code (only when download=True).

    Example:
        >>> # Actually download
        >>> file_path = download_s3_file("https://example.com/file.txt", save_path="/data")
        >>> print(file_path)
        /data/file.txt
        
        >>> # Just get expected path
        >>> expected_path = download_s3_file("https://example.com/file.txt", save_path="/data", download=False)
        >>> print(expected_path)
        /data/file.txt
    """    
    parsed_url = urlparse(file_url)
    file_name = os.path.basename(parsed_url.path.split('?')[0])  # Remove query parameters
    
    if save_path:
        if os.path.isdir(save_path) or save_path.endswith('/'):
            local_file_path = os.path.join(save_path, file_name)
        else:
            local_file_path = save_path
    else:
        local_file_path = os.path.join(tmp_dir, file_name)

    # If download=False, just return the expected path
    if not download:
        return local_file_path

    # Check if file exists and we shouldn't overwrite
    if os.path.exists(local_file_path):
        return local_file_path

    # Actually download the file
    response = requests.get(file_url, stream=True)
    if response.status_code == 200:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
        
        with open(local_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download file: {response.status_code}")

    return local_file_path

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


def adapt_columns_for_dpo_dataset(dataset_path: str, dataset_type: DpoDatasetType, apply_formatting: bool = False):
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

    print("Transformed dataset to include chatml.intel field names:")
    print(f"Final fields: {list(output_data[0].keys()) if output_data else []}")
    print(f"Dataset saved to {dataset_path}")


def adapt_columns_for_grpo_dataset(dataset_path: str, dataset_type: GrpoDatasetType):
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
    # Remove records where the prompt field is empty or None
    df = df[df[cst.GRPO_DEFAULT_FIELD_PROMPT].notna() & (df[cst.GRPO_DEFAULT_FIELD_PROMPT] != "")]
    output_data = df.to_dict(orient='records')
    with open(dataset_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Transformed dataset to adapt to axolotl's `{cst.GRPO_DEFAULT_FIELD_PROMPT}` expected column name.")
