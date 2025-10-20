import os
import json
import argparse
from pathlib import Path
from juristiq.inference.prompts import (
    get_annot_output,
    get_judge_evaluation_prompt
)
from juristiq.inference.models import ModelName
from juristiq.data_preprocessing.annots import load_data_from_jsonl


def _validate_args(args: argparse.Namespace) -> None:
    """
    Validate the command line arguments.

    Args:
        args: The command line arguments.

    Raises:
        argparse.ArgumentError: If any of the arguments is invalid.
    """
    if not os.path.exists(args.input_file):
        raise argparse.ArgumentError(args.input_file, f"The input file '{args.input_file}', containing the annotation samples, does not exist.")
    
    output_path = Path(args.output_file)
    if output_path.exists() and not output_path.is_file():
        raise argparse.ArgumentError(args.output_file, f"The output path '{args.output_file}' is not a file.")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Creates evaluation dataset for calculating baseline metrics.")

    parser.add_argument("-i", "--input-file",
                         required=True,
                         help="Local file system path to a .jsonl file containing the annotation samples.")
    parser.add_argument("-o", "--output-file", 
                        required=True, 
                        help="Local file system path to a .jsonl file where the data, in the evaluation format, will be saved.")
    parser.add_argument("-m", "--model-id",
                        required=True,
                        help="The identifier of the model that was used for the batch inference.")

    args = parser.parse_args()

    _validate_args(args)

    return args
    

def main(input_file_path: str,
         output_file_path: str,
         model_id: str):
    """
    Creates and saves the evaluation dataset to an output file.

    Args:
        input_file_path: a path to the annotation samples file.
        output_file_path: a path to the file containing the evaluation dataset.
        model_id: an identifier of the model that will be used for inference.
    """
    annots = load_data_from_jsonl(input_file_path)
    model_name = ModelName(model_id)
    
    with open(output_file_path, mode="w", encoding="utf8") as fp:
        for annot in annots:
            
            record = {
                "prompt": get_judge_evaluation_prompt(model_name, annot),
                "referenceResponse": get_annot_output(annot)
            }

            fp.write(json.dumps(record) + "\n")


if __name__ == "__main__":

    args = parse_args()
    _validate_args(args)

    main(args.input_file, 
         args.output_file,
         args.model_id)
