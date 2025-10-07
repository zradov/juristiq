import os
import json
import logging
import argparse
from pathlib import Path
from annots_utils import get_hash
from genai_utils import log_user_balance
from typing import Generator, Tuple, Union
from logging_config import configure_logging
from genai_exceptions import (
    GenAIAuthError,
    GenAIInsufficientBalanceError
)
from data_providers import get_data_provider, DataProvider
from openai.types.chat.chat_completion import ChatCompletion
from llm_utils import get_contract_review_prompt_builder, get_genai_client
from genai_clients import GenAIClient, GenAIClientFactory, GenAIChatMetrics


configure_logging()
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parses and returns CLI options.

    Returns:
        argparse.Namespace object containing the CLI options and values.
    """
    parser = argparse.ArgumentParser(description="CUAD Annotations Review")
    parser.add_argument("-t", "--transformed-cuad-annots-path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket containing transformed CUAD annotations.")
    parser.add_argument("-r", "--reviewed-cuad-annots-path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket containing the reviewed CUAD annotations.")
    parser.add_argument("-p", "--policies-path", 
                        type=str, 
                        required=True, 
                        help="Path to the file containing law documents compliance policies.")
    parser.add_argument("-g", "-genai-provider",
                        type=str,
                        required=False,
                        default="deep_seek",
                        help="The name of the GenAI provider, available: deep_seek")
    args = parser.parse_args()

    return args


def load_annots(data_provider: DataProvider, path: str) -> Generator[Tuple[str, dict], None, None]:
    """
    Loads all annotations files in the specified location.

    Args:
        data_provider: an instance of the DataProvider class.
        path: a location to annotations folder.

    Returns:
        yields a tuple containing with the annotations file path and the JSON document 
        with parsed file's content.
    """
    for annots_file_name in data_provider.list_objects(path):
        annots = data_provider.get_object(os.path.join(path, annots_file_name))
        annots = json.loads(annots)

        yield Path(path) / annots_file_name, annots


def load_policies(policies_path: str, 
                  data_provider: DataProvider) -> dict:
    """
    Loads and returns compliance policies.

    Args:
        policies_path: a path to the file containing compliance policies.
        data_provider: an instance of the DataProvider class.

    Returns:
        a JSON document representing the compliance policies.
    """
    policies_path = Path(policies_path)
    policies = json.loads(data_provider.get_object(policies_path))

    return policies


def log_chat_metrics(chat_metrics: GenAIChatMetrics) -> None:
    """
    Logs the response metrics of the chat completion request.

    Args:
        chat_metrics: an instance of the GenAIChatMetrics class.
    """
    logger.info((f"Chat metrics: completion_tokens={chat_metrics.completion_tokens}, prompt_tokens="
                 f"{chat_metrics.prompt_tokens}, total_tokens={chat_metrics.total_tokens}, "
                 f"cached_tokens={chat_metrics.prompt_tokens_details.cached_tokens}, "
                 f"prompt_cache_hit_tokens={chat_metrics.prompt_cache_hit_tokens}, "
                 f"prompt_cache_miss_tokens={chat_metrics.prompt_cache_miss_tokens}"
    
    ))


def parse_llm_output(output: str) -> dict:
    """
    Parses the response of the LLM request into a structured dictionary
    where keys and values are separated by the ":" colon character in 
    the output string.

    Args:
        output: chat completion response.

    Returns:
        a structured dictionary
    """
    parsed_output = {}
    for line in output.split("\n"):
        parts = line.split(":")
        key, value = parts[0], ":".join(parts[1:])
        annot_key = "_".join(key.strip().lower().split(" "))
        parsed_output[annot_key] = value.strip()

    return parsed_output


def process_annot_item(item: dict, 
                       prompt_hash: str, 
                       annots_file_path: Path,
                       client: GenAIClient,
                       messages: list[dict]) ->  Tuple[dict, ChatCompletion]:
    if not item["context"].strip():
        logger.info((f"Missing context, prompt with hash {prompt_hash},"
                        f"in annotations file {annots_file_path.name}, skipped."))
        parsed_output = {
            "review_label": "Missing",
            "suggested_redline": "N/A",
            "rationale": "Contract excerpt is not provided for review."
        }
        return parsed_output, None
    
    response = client.send_request(messages=messages)
    chat_metrics = client.get_chat_metrics()
    log_chat_metrics(chat_metrics)
    content = response.choices[0].message.content
    logger.info(f"Content: {content}")
    parsed_output = parse_llm_output(content)
    
    return parsed_output, response


def print_stats(index: int, 
                total_tokens: int,
                cache_hit_tokens: int) -> None:
    print()
    logger.info(f"Processed {index+1} annotations.")
    logger.info(f"Total tokens: {total_tokens}, cache hit tokens: {cache_hit_tokens}.")
    log_user_balance()
    print()


def review_annots(transformed_cuad_annots_path: str,
                  reviewed_cuad_annots_path: str,
                  policies_path: Union[str, Path],
                  genai_provider_name: str) -> None:
    """
    Runs compliance review on all transformed CUAD annotations and stores
    the augmented annotations in the specified locations.

    Args:
        transformed_cuad_annots_path: a location of the transformed CUAD annotations.
        reviewed_cuad_annots_path: a location of the reviewed CUAD annotations.
        policies_path: a location of the compliance policies file.
        genai_provider_name: a name of the GenAI provider.
        
    Returns:
        None
    """
    client = GenAIClientFactory.create(genai_provider_name)
    data_provider_obj = get_data_provider(transformed_cuad_annots_path)
    prompt_builder = get_contract_review_prompt_builder()
    policies = load_policies(policies_path)
    log_user_balance()
    existing_reviews = data_provider_obj.list_objects(reviewed_cuad_annots_path)
    index = 1
    total_tokens = 0
    cache_hit_tokens = 0
    
    for annots_file_path, annots in load_annots(data_provider_obj, transformed_cuad_annots_path):
        reviews_annots_dir = Path(reviewed_cuad_annots_path) / annots_file_path.stem.strip()
        data_provider_obj.make_containers(reviews_annots_dir)
        logger.info(f"Processing transformed annotations for {annots_file_path.name} ...")
        try:
            for item in annots["qas"]:
                messages = prompt_builder(item=item, policies=policies)
                prompt_hash = get_hash(messages)
                new_review_file_name = f"{prompt_hash}.json"
                if new_review_file_name in existing_reviews:
                    logger.info(f"Annotations for file {prompt_hash}.json already exists.")
                    continue
                parsed_output, response = process_annot_item(item, 
                                                             prompt_hash, 
                                                             annots_file_path,
                                                             client,
                                                             messages)
                if response:
                    total_tokens += response.usage.total_tokens
                    cache_hit_tokens += response.usage.prompt_cache_hit_tokens
                output_item = item.copy()
                output_item.update(parsed_output)
                data_provider_obj.put_object(reviews_annots_dir / new_review_file_name, json.dumps(output_item))
                if index % 100 == 0:
                    print_stats(index, total_tokens, cache_hit_tokens) 
                index += 1
        except GenAIAuthError as err:
            logger.error(f"Authentication failed. {err}.")
            raise
        except GenAIInsufficientBalanceError as err:
            logger.error(f"Insufficient balance in the user account for making the request. {err}.")
            log_user_balance(genai_provider_name)
            raise
        except Exception as err:
            logger.error(f"Failed to oversample the annotations. {err}.")
            raise
