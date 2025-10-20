import os
import argparse
from typing import List, Dict
from dotenv import load_dotenv
from juristiq.inference.models import ModelName
from juristiq.cloud.utils import get_bedrock_runtime_client
from juristiq.inference.prompts import ModelTokenizer
from juristiq.data_preprocessing.annots import load_data_from_jsonl
from juristiq.inference.prompts import (
    get_system_message, 
    get_user_message, 
    get_annot_output,
    get_system_prompt_text
)
from juristiq.inference.tokens_counter import TokensCounter
from juristiq.monitoring.costs_calculator import (
    calculate_costs, 
    ProcessingMode, 
    TokensMetrics
)
from juristiq.config.consts import TOKENS_RATIO_FILE_PATH


def  _validate_args(parser: argparse.ArgumentParser, 
                    args: argparse.Namespace):
    
    if not os.path.exists(args.input_file):
        parser.error(f"The input file '{args.input_file}' does not exist.")
    
    if not os.path.isfile(args.input_file):
        parser.error(f"The input file '{args.input_file}' is not a file.")
    
    if args.model_id not in ModelName:
        parser.error(f"The model id '{args.model_id}' is not supported.")
    

def _parse_args():

    parser = argparse.ArgumentParser(description="Bedrock Batch Inference Job Creation")

    parser.add_argument("-i", "--input-file",
                        required=False,
                        type=str,
                        help="A local file system path to the .jsonl file containing the annotations samples.")
    parser.add_argument("-m", "--model-id",
                        required=False,
                        default=ModelName.NOVA_LITE.value,
                        help="A base model id of the Bedrock LLM that will be used for calculating the tokens count.")
    parser.add_argument("-u", "--update-tokens-ratios",
                        action=argparse.BooleanOptionalAction,
                        help="Whether or not update tokens ratios by running the inference using Bedrock models on a subset of prompt samples.")


    args = parser.parse_args()

    _validate_args(parser, args)

    return args


def _get_prompt_input(system_message: Dict, annot: Dict) -> Dict:

    user_message = get_user_message(annot)
    prompt_input = {**system_message, "messages": [user_message]}

    return prompt_input


def _get_prompts(input_file_path: str,
                 model_id: str) -> List[Dict]:

    annots = load_data_from_jsonl(input_file_path)
    system_prompt = get_system_prompt_text(ModelName(model_id))
    system_message = get_system_message(system_prompt)
    prompt_inputs = []
    prompt_outputs = []

    for annot in annots:
        prompt_inputs.append(_get_prompt_input(system_message, annot))
        prompt_outputs.append(get_annot_output(annot))

    return prompt_inputs, prompt_outputs


def _count_tokens(prompts: List[Dict],
                  tokens_counter: TokensCounter) -> int:
    
    total_tokens = 0

    for prompt in prompts:
        total_tokens += tokens_counter.count_tokens(prompt)

    return total_tokens


def main(input_file_path: str, 
         model_id: str):
    
    model_name = ModelName(model_id)
    prompt_inputs, prompt_outputs = _get_prompts(input_file_path, model_id)
    tokens_counter = TokensCounter.create(model_name)
    input_tokens_count = _count_tokens(prompt_inputs, tokens_counter)
    output_tokens_count = _count_tokens(prompt_outputs, tokens_counter)
    tokens_metrics = TokensMetrics(total_input_tokens=input_tokens_count,
                                   total_output_tokens=output_tokens_count)
    on_demand_token_costs = calculate_costs(model_name, tokens_metrics)
    batch_inference_token_costs = calculate_costs(model_name, tokens_metrics, processing_mode=ProcessingMode.BATCH)
    
    print(f"Input tokens count: {input_tokens_count}.")
    print(f"Output tokens count: {output_tokens_count}")
    print(f"Total on-demand cost for input: {on_demand_token_costs.input:.2f}")
    print(f"Total on-demand cost for output: {on_demand_token_costs.input:.2f}")
    print(f"Total batch inference cost for input: {batch_inference_token_costs.input:.2f}")
    print(f"Total batch inference cost for output: {batch_inference_token_costs.input:.2f}")
    

if __name__ == "__main__":

    load_dotenv()

    args = _parse_args()

    main(args.input_file, args.model_id)
