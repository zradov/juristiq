import os
import json
from argparse import (
    ArgumentParser, 
    Namespace
)
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
from juristiq.file.utils import FileWriter
from juristiq.config.inference import DEFAULT_BATCH_INFERENCE_PARAMS


def _validate_args(parser: ArgumentParser,
                   args: Namespace) -> None:
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
    parser = ArgumentParser(description="Full Stratified Split + JSONL Conversion")

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
    system_prompt = Path(system_prompt_path).read_text(encoding="utf8")
    config = _get_inference_config(system_prompt)
    
    with open(annots_path, mode="r", encoding="utf8") as input_fp:
        with FileWriter(output_path.name, output_path.parent, max_file_size) as fw:
            for line in input_fp.readlines():
                annot = json.loads(line)
                annot_text = get_annot_input(annot)
                record = get_batch_inference_record(annot_text, annot["hash"], ModelType.NOVA, config)
                fw.write(record)


if __name__ == "__main__":

    args = parse_args()

    print(f"AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT: {AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT}")

    create_dataset(args.annots_path, 
                   AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT,
                   args.output_file,
                   args.max_file_size)
