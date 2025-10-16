import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Tuple, List, Dict
from ...config.logging import configure_logging


configure_logging()
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Full Stratified Split + JSONL Conversion")

    parser.add_argument("-a", "--annots-path",
                         required=True,
                         help="A local file system path to the folder containing the annotations.")
    parser.add_argument("-o", "--output-dir", 
                        required=True, 
                        help="A local file system path to the folder where the training, validation and the test datasets should be stored.")
    parser.add_argument("-t", "--train-pct", 
                        default=0.8, 
                        type=float, 
                        help="The percentage of the data that is split into the training subset.")
    parser.add_argument("-v", "--val-pct", 
                        default=0.15, 
                        type=float, 
                        help="The percentage of the data that is split into the validation subset.")
    parser.add_argument("-i", "--include-answers",
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    return args


def validate_args(args):
    if not os.path.exists(args.annots_path):
        raise argparse.ArgumentError(args.annots_path, f"The annotations path '{args.annots_path}' does not exist.")
    if args.train_pct < 0 or args.train_pct > 1.0:
        raise argparse.ArgumentError(args.train_pct, f"The training percentage '{args.train_pct}' must be within [0, 1] range.")
    if args.val_pct < 0 or args.val_pct > 1.0:
        raise argparse.ArgumentError(args.val_pct, f"The validation percentage '{args.val_pct}' must be within [0, 1] range.")
    if args.train_pct + args.val_pct > 1.0:
        raise argparse.ArgumentError(None, f"The sum of the training and the validation percentage must be withing the [0, 1] range.")     


def load_annots(annots_path: str) -> Dict[str, List]:

    annots = defaultdict(list)

    for file_path in Path(annots_path).rglob("*.json"):
        with open(file_path, mode="r", encoding="utf8") as fp:
            data = json.load(fp)
            key = f"{data['review_label']}::{data['clause_type']}"
            annots[key].append(data)

    return dict(annots)


def split_annots(annots: dict[str, List],
                 train_pct: float,
                 val_pct: float) -> Tuple[List[dict], List[dict], List[dict]]:
    
    train_annots = [] 
    val_annots = [] 
    test_annots = []

    for values in annots.values():
        values_count = len(values)
        train_count = int(values_count * train_pct)
        val_count = int(values_count * val_pct)
        train_annots.extend(values[:train_count])
        val_annots.extend(values[train_count:train_count+val_count])
        test_annots.extend(values[train_count+val_count:])

    return train_annots, val_annots, test_annots


def get_file_record(annot: Dict, include_answers: bool=False) ->  str:

    input_parts = [
        f"Question: {annot['question']}",
        f"Context: {annot['context']}",
        f"Policy: {annot['policy_text']}",
        f"Clause Type: {annot['clause_type']}"
    ]

    if include_answers and annot["answers"]:
        input_parts.insert(1, f"Answers: {' '.join(annot['answers'])}")

    record = {
        "input": "\n".join(input_parts),
        "output": (
            f"Review Label: {annot['review_label']}\n"
            f"Suggested redline: {annot['suggested_redline']}\n"
            f"Rationale: {annot['rationale']}"
        )
    }

    return json.dumps(record, ensure_ascii=False)


def save(annots: List[Dict], 
         output_dir: str, 
         file_name: str,
         include_answers: bool) -> None:

    save_path = Path(output_dir) / file_name
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, mode="w", encoding="utf8") as fp:
        for annot in annots:
            fp.write(get_file_record(annot, include_answers=include_answers) + "\n")


def main(annots_path: str, 
         output_dir: str, 
         train_pct: float, 
         val_pct: float,
         include_answers: bool):
    
    all_annots = load_annots(annots_path)

    train_annots, val_annots, test_annots = split_annots(all_annots, train_pct, val_pct)

    for annots, dest_file_name in [(train_annots, "train.jsonl"), 
                                   (val_annots, "val.jsonl"),
                                   (test_annots, "test.jsonl")]:
        save(annots, output_dir, dest_file_name, include_answers)


if __name__ == "__main__":
    try:
        args = parse_args()
        validate_args(args)
        main(args.annots_path, 
             args.output_dir, 
             args.train_pct, 
             args.val_pct,
             bool(args.include_answers))
    except SystemExit:
        sys.exit(1)
