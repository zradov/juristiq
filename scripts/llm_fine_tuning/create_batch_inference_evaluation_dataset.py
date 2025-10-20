import os
import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Creates evaluation dataset for calculating baseline metrics.")

    parser.add_argument("-i", "--input-file",
                         required=True,
                         help="Local file system path to a .jsonl file containing the output results produced after running the batch inference.")
    parser.add_argument("-g", "--ground-truth-file",
                        required=True,
                        help="Local file system path to a .jsonl file containing the ground truth responses.")
    parser.add_argument("-o", "--output-file", 
                        required=True, 
                        help="Local file system path to a .jsonl file where the data, in the evaluation format, will be saved.")
    parser.add_argument("-m", "--model-name",
                        required=True,
                        help="The name of the model that was used for the batch inference.")

    args = parser.parse_args()

    return args
    

def _validate_args(args: argparse.Namespace) -> None:
    """
    Validate the command line arguments.

    Args:
        args: The command line arguments.

    Raises:
        argparse.ArgumentError: If any of the arguments is invalid.
    """
    if not os.path.exists(args.input_file):
        raise argparse.ArgumentError(args.input_file, f"The input file '{args.input_file}', containing the batch inference results, does not exist.")
    
    if not os.path.exists(args.ground_truth_file):
        raise argparse.ArgumentError(args.ground_truth_file, f"The ground truth file '{args.ground_truth_file}' does not exist.")
    
    output_path = Path(args.output_file)
    if output_path.exists() and not output_path.is_file():
        raise argparse.ArgumentError(args.output_file, f"The output path '{args.output_file}' is not a file.")
    
    if args.max_file_size <= 0:
        raise argparse.ArgumentError(args.max_file_size, f"The maximum file size '{args.max_file_size}' must be a positive integer.")


def _load_data(file_path: str) -> dict:

    data = []

    with open(file_path, mode="r", encoding="utf8") as fp:
        for line in fp:
            data.append(json.loads(line))

    return data


def _get_ground_truth_response(annot: dict) -> str:

    return (
        f"Review label: {annot['review_label']}. "
        f"Rationale: {annot['rationale']} "
        f"Suggested Redline: {annot['suggested_redline']}"
    )


def main(input_file_path: str,
         ground_truth_file_path: str,
         output_file_path: str,
         model_name: str):
    """
    Creates and saves the evaluation dataset to an output file.

    Args:
        input_file_path: a path to the batch inference output file.
        ground_truth_file: a path to the file containing ground truth response that 
                           the evaluation model can reference during the evaluation.
        output_file_path: a path to the file containing the evaluation dataset.
        model_name: the name of the batch inference model.
    """
    input_data = _load_data(input_file_path)
    ground_truth_data = _load_data(ground_truth_file_path)
    ground_truth_data = { i["hash"]: i for i in ground_truth_data }

    with open(output_file_path, mode="w", encoding="utf8") as fp:
        for input_item in input_data:
            model_output = input_item.get("modelOutput", {}).get("results", [{}])[0].get("outputText", "")

            if model_output:
                output_item = {
                    "prompt": input_item["modelInput"]["inputText"],
                    "modelResponses": [{
                        "response": model_output,
                        "modelIdentifier": model_name
                    }],
                    "referenceResponse": _get_ground_truth_response(ground_truth_data[input_item["recordId"]])
                }
            fp.write(json.dumps(output_item) + "\n")


if __name__ == "__main__":

    args = parse_args()
    _validate_args(args)

    main(args.input_file, 
         args.ground_truth_file, 
         args.output_file,
         args.model_name)
