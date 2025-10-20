import json
import random
import argparse
from juristiq.inference.models import ModelName
from typing import List, Dict
from juristiq.inference.tokens_counter import TokensCounter
from juristiq.cloud.utils import get_bedrock_runtime_client
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


def _update_tokens_ratios(model_name: ModelName,
                          prompt_inputs: List[Dict],
                          prompt_outputs: List[Dict],
                          max_samples: int=20,
                          samples_pct: float=0.01) -> None:
    
    samples = []

    for inputs in [prompt_inputs, prompt_outputs]:
        samples_count = min(int(len(inputs)*samples_pct), max_samples)
        samples.extend(random.sample(inputs, samples_count)) 

    if samples:
        client = get_bedrock_runtime_client()
        tokens_counter = TokensCounter.create(model_name)
        ratios = []
        
        for sample in samples:
            if "messages" not in sample:
                payload = {
                    "messages": [{"role": "user", "content": [{"text": sample}]}],
                    "inferenceConfig": {
                        "maxTokens": 1
                    }
                }
                res = client.invoke_model(modelId=model_name.value,
                                          contentType="application/json",
                                          body=json.dumps(payload))
                res = json.loads(res["body"].read())
            else:
                res = client.converse(modelId=model_name.value,
                                      messages=sample["messages"],
                                      system=sample["system"],
                                      inferenceConfig={"maxTokens": 1})
            inference_input_tokens_count = res["usage"]["inputTokens"]
            input_tokens_count = tokens_counter.count_tokens(sample)
            ratios.append(input_tokens_count/inference_input_tokens_count)
        
        data = {"tokenizers": {model_name.name: sum(ratios)/len(ratios)}}
        Path(TOKENS_RATIO_FILE_PATH).write_text(json.dumps(data), encoding="utf8")
    