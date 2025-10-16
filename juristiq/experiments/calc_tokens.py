import os
import json
from typing import Generator, Tuple
from transformers import AutoTokenizer
from llm.utils import get_contract_review_prompt_builder

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id)


def load_file(path: str) -> str:
    
    with open(path, encoding="utf-8") as fp:
        return fp.read()


def load_annots(path: str) -> Generator[Tuple[str, dict], None, None]:
    
    for annots_file_name in os.listdir(path):
        annots_file_path = os.path.join(path, annots_file_name)
        annots = load_file(annots_file_path)
        annots = json.loads(annots)
        yield annots_file_name, annots


policies_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "data", "policies.json")
annots_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "output", "juristiq-cuad-transformed")
policies = json.loads(load_file(policies_path))
prompt_builder = get_contract_review_prompt_builder()

tokens_count = 0
total_requests = 0

for i, (annots_file_name, annots) in enumerate(load_annots(annots_path)):
    print(f"Processing {annots_file_name} ...")
    for item in annots["qas"]:
        chars_count = 0
        temp_tokens_count = 0
        messages = prompt_builder(item, policies)
        for msg_type, msg in messages.items():
            chars_count += len(msg["content"])
            temp_tokens_count += len(tokenizer.encode(msg["content"]))
        print(f"  Characters count: {chars_count}")
        print(f"  Tokens count: {temp_tokens_count}")
        total_requests += 1
        tokens_count += temp_tokens_count

    if i % 20:
        print(f"Processed {i+1} annotation files.")

print(f"Total tokens: {tokens_count}")
print(f"Total requests: {total_requests}")

