import os
import json
from argparse import (
    ArgumentParser, 
    ArgumentTypeError
)
from pathlib import Path
from juristiq.inference.prompts import (
    get_annot_output,
    get_evaluation_inference_prompt
)
from juristiq.file.utils import FileWriter
from juristiq.inference.models import ModelName
from juristiq.data_preprocessing.annots import load_data_from_jsonl


def input_file_path(value):

    if not os.path.exists(value):
        raise ArgumentTypeError(f"The input file '{value}', containing the annotation samples, does not exist.")

    return value


def output_folder_path(value):

    output_path = Path(value)
    if output_path.exists() and output_path.is_file():
        raise ArgumentTypeError(f"The output folder '{value}' is not a folder.")

    return value


def model_id(value):

    if value not in ModelName:
        raise ArgumentTypeError(f"Unsupported model id: {model_id}")

    return value


def max_file_size(value):

    val = int(value)

    if val <= 0:
        raise ArgumentTypeError(f"The maximum file size '{value}' must be a positive integer.")

    return val


def parse_args():
    parser = ArgumentParser(description="Creates evaluation dataset for calculating baseline metrics.")

    parser.add_argument("-i", "--input-file",
                         required=True,
                         type=input_file_path,
                         help="Local file system path to a .jsonl file containing the annotation samples.")
    parser.add_argument("-o", "--output-folder", 
                        required=True,
                        type=output_folder_path,
                        help="Local file system path to a folder where the files, in the evaluation format, will be saved.")
    parser.add_argument("-m", "--model-id",
                        required=True,
                        type=model_id,
                        help="The identifier of the model that will be used for inference.")
    parser.add_argument("-s", "--max_file_size",
                        required=False,
                        type=max_file_size,
                        default=100,
                        help="A maximum file size in MB of a .jsonl file, if surpassed, another file is created.")
    
    args = parser.parse_args()

    return args
    

def main(input_file_path: str,
         output_folder_path: str,
         model_id: str,
         max_file_size: int):
    """
    Creates and saves the evaluation dataset to an output file.

    Args:
        input_file_path: a path to the annotation samples file.
        output_file_path: a path to the file containing the evaluation dataset.
        model_id: an identifier of the model that will be used for inference.
        max_file_size (int): Maximum size of each output file in MB.
    """
    annots = load_data_from_jsonl(input_file_path)
    model_name = ModelName(model_id)
    
    with FileWriter(input_file_path, output_folder_path, max_file_size) as fw:
        for annot in annots:    
            record = {
                "prompt": get_evaluation_inference_prompt(model_name, annot),
                "referenceResponse": get_annot_output(annot)
            }
            fw.write(json.dumps(record) + "\n")


if __name__ == "__main__":

    args = parse_args()

    main(args.input_file, 
         args.output_folder,
         args.model_id,
         args.max_file_size)
