import os
import json
import argparse
from io import StringIO
from pathlib import Path
from juristiq.config.templates import (
    AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT
)
from juristiq.inference.prompts import (
    BaseGenerationConfig,
    ModelType,
    get_batch_inference_record,
    get_annot_input
)
from juristiq.config.inference import DEFAULT_BATCH_INFERENCE_PARAMS


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    """
    Validate the command line arguments.

    Args:
        args: The command line arguments.

    Raises:
        argparse.ArgumentError: If any of the arguments is invalid.
    """
    if not os.path.exists(args.annots_path):
        parser.error(f"The annotations path '{args.annots_path}' does not exist.")
    
    output_path = Path(args.output_file)
    if output_path.exists() and not output_path.is_file():
        parser.error(f"The output path '{args.output_file}' is not a file.")
    
    if args.max_file_size <= 0:
        parser.error(f"The maximum file size '{args.max_file_size}' must be a positive integer.")


def parse_args():
    parser = argparse.ArgumentParser(description="Full Stratified Split + JSONL Conversion")

    parser.add_argument("-a", "--annots-path",
                         required=True,
                         help="A local file system path to the folder containing the annotations.")
    parser.add_argument("-o", "--output-file", 
                        required=True, 
                        help="A local file system path to a file where the data will be saved.")
    parser.add_argument("-m", "--max_file_size",
                        required=False,
                        default=100,
                        help="A maximum file size in MB of a .jsonl file, if surpassed, another file is created.")
    
    args = parser.parse_args()

    _validate_args(parser, args)

    return args
    

def _get_inference_config(system_prompt: str) -> BaseGenerationConfig:
    """
    Get the inference configuration for batch processing.

    Args:
        system_prompt (str): The system prompt to be used in the configuration.
    
    Returns:
        BaseGenerationConfig: The configuration object for batch inference.
    """
    return BaseGenerationConfig(
        **DEFAULT_BATCH_INFERENCE_PARAMS.model_dump(),
        system=system_prompt
    )


def _get_unique_path(path: Path, 
                     file_stem: str) -> Path:
    """
    Generate a unique file path by appending an index if the file already exists.

    Args:
        path (Path): The original file path.
        file_stem (str): The stem of the file name to be used for indexing.

    Returns:
        a path object representing a unique file path.
    """
    if path.exists():
        file_index = len(list(path.parent.rglob("*.jsonl"))) + 1
        path = path.parent / f"{file_stem}{file_index}{path.suffix}"

    return path


def _write_text(file_path: Path, file_stem: str, buffer: StringIO) -> None:
    """
    Write the contents of the buffer to a uniquely named file.

    Args:
        file_path (Path): The base file path where the buffer will be written.
    """
    output_path = _get_unique_path(file_path, file_stem)
    output_path.write_text(buffer.getvalue(), encoding="utf8")
                

def create_dataset(annots_path: str, 
                   system_prompt_path: str, 
                   output_file: str,
                   max_file_size: int) -> None:
    """
    Create a batch inference dataset from the given annotations.

    Args:
        annots_path (str): Path to the annotations file.
        system_prompt_path (str): Path to the system prompt file.
        output_file (str): Path to the output file where the dataset will be saved.
        max_file_size (int): Maximum size of each output file in MB.

    Raises:
        FileNotFoundError: If the annotations file or system prompt file does not exist.
    """
    for path in [annots_path, system_prompt_path]:
        if not os.path.exists(annots_path):
            raise FileNotFoundError(f"File {path} does not exist.")
    
    output_path = Path(output_file)
    _ = [p.unlink() for p in output_path.parent.rglob("*.jsonl")]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file_stem = output_path.stem
    # the existing file will be deleted before saving the new data.
    max_file_size_in_bytes = max_file_size * 1024 * 1024
    print(f"max_file_size_in_bytes: {max_file_size_in_bytes}")

    system_prompt = Path(system_prompt_path).read_text(encoding="utf8")
    config = _get_inference_config(system_prompt)
    
    with open(annots_path, mode="r", encoding="utf8") as input_fp:
        buffer = StringIO()
        current_buffer_size = 0

        for idx, line in enumerate(input_fp.readlines()):
            annot = json.loads(line)
            annot_text = get_annot_input(annot)
            record = get_batch_inference_record(annot_text, annot["hash"], ModelType.NOVA, config)
            record_size = len(record.encode("utf8"))
            if current_buffer_size + record_size > max_file_size_in_bytes:
                _write_text(output_path, output_file_stem, buffer)
                buffer.truncate(0)
                buffer.seek(0)
                current_buffer_size = 0
            buffer.write(record)
            current_buffer_size += record_size
        
        if buffer.tell() != 0:
            _write_text(output_path, output_file_stem, buffer)


if __name__ == "__main__":

    args = parse_args()

    print(f"AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT: {AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT}")

    create_dataset(args.annots_path, 
                   AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT,
                   args.output_file,
                   args.max_file_size)
