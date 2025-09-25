import re
import os
import json
import config
import logging
import argparse
import logging_config
from multiprocessing import Pool
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from data_provider import (
    DataProvider,
    LocalDataProvider, 
    S3DataProvider
)
from langchain_community.utils.math import cosine_similarity
from langchain_experimental.text_splitter import SemanticChunker


file_log_handler = logging.FileHandler(filename=logging_config.LOG_PATH, encoding="utf-8")
file_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
console_log_handler = logging.StreamHandler()
console_log_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
logging.basicConfig(handlers=[file_log_handler,  console_log_handler], level=logging.INFO)
logger = logging.getLogger(__name__)

CLAUSE_TYPE_PATTERN = ".*related to \"(?P<clause_type>[^\"]+)\".*"

_data_provider = None


def parse_args():
    parser = argparse.ArgumentParser(description="CUAD Utilities")
    parser.add_argument("-c", "--chunks_annots_path", type=str, required=True, help="Path to a local folder or to the S3 bucket containing chunked CUAD annotations.")
    parser.add_argument("-f", "--full_contracts_path", type=str, required=True, help="Path to a local folder or to the S3 bucket containing full contracts directory.")
    parser.add_argument("-p", "--policies-path", type=str, required=True, help="Path to the file containing law documents compliance policies.")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Path to a local folder or to the S3 bucket where transformed CUAD annotations will be stored.")
    args = parser.parse_args()

    return args


def find_chunk_limits(contract_text: str, 
                      chunk: str, 
                      offset: int,
                      min_chunk_len: int=40) -> tuple[int, int]:
    """
    Find the approximate start and end positions of a chunk within the contract text.

    Args:
        contract_text (str): The full text of the contract.
        chunk (str): The text chunk to locate within the contract.
        offset (int): The position in the contract text to start searching from.
        min_chunk_len (int): Minimum length of chunk to consider for searching.

    Returns:
        A tuple containing the start and end positions of the chunk within the contract text. 
        Returns (-1, -1) if the chunk is not found.
    """

    if len(chunk) < min_chunk_len:
        return -1, -1
    start = contract_text.find(chunk, offset)
    if start == -1:  
        half_chunk_len = len(chunk)//2
        chunk1 = chunk[:half_chunk_len]
        chunk2 = chunk[half_chunk_len:]
        start1, end1 = find_chunk_limits(contract_text, chunk1, offset)
        if start1 != -1:
            start = start1
        else:
            start2, end2 = find_chunk_limits(contract_text, chunk2, offset)
            if start2 != -1:
                start = start2 - len(chunk1)
            else:
                return -1, -1
    end = start + len(chunk)
    return start, end


def split_contract(contract_text: str) -> list[dict]:
    embed_model = HuggingFaceEmbeddings(model_name=config.CONTRACTS_TEXT_EMBEDDINGS_MODEL)
    # the 'gradient' method works best for legal documents.
    chunker = SemanticChunker(embeddings=embed_model, breakpoint_threshold_type="gradient")
    chunks = []
    offset = 0

    for chunk in chunker.split_text(contract_text):
        start, end = find_chunk_limits(contract_text, chunk, offset)
        chunks.append({
            "text": chunk, 
            "start_pos": start, 
            "end_pos": end
        })
        offset = end

    return chunks


def get_clause_type(clause):
    match = re.match(CLAUSE_TYPE_PATTERN, clause)
    if match:
        return match.group("clause_type")
    return None


def map_answers_to_chunks(answers: list[dict], contract_chunks: list[dict]) -> list[dict]:
    matched_chunks = []

    for answer in answers:
        for chunk in contract_chunks:
            answer_start = answer["answer_start"]
            if chunk["start_pos"] <= answer_start < chunk["end_pos"]:
                if chunk["text"] not in matched_chunks:
                    matched_chunks.append(chunk["text"])
                    break

    return matched_chunks


def process_cuad_questions(questions: list[dict], 
                           contract_chunks: list[dict],
                           policies: dict) -> list[dict]:
    processed_questions = []

    for question in questions:
        clause_type = get_clause_type(question["question"])
        if not clause_type:
            logger.warning(f"Could not extract clause type from question: {question['question']}")
            continue
        matched_chunks = map_answers_to_chunks(question["answers"], contract_chunks)
        processed_questions.append({
            "question": question["question"],
            "answers": [ans["text"] for ans in question["answers"]],
            "context": " ".join(matched_chunks),
            "clause_type": clause_type,
            "review_label": "TBD",
            "policy_id": policies[clause_type]["policy_id"],
            "policy_text": policies[clause_type]["policy_text"],
            "suggested_readline": "TBD",
            "rationale": "TBD"
        })

    return processed_questions


def process_cuad_paragraphs(data_item: dict, 
                            contract_chunks: list[dict],
                            policies: dict) -> list[dict]:
    
    all_entries = []

    for paragraph in data_item["paragraphs"]:
        questions_entries = process_cuad_questions(paragraph["qas"], contract_chunks, policies)
        if questions_entries:
            all_entries.extend(questions_entries)

    return all_entries


def load_chunk_annots(chunk_annots_path, contract_name) -> dict:

    contract_annots = _data_provider.get_object(chunk_annots_path, contract_name)
    return json.loads(contract_annots)
    

def load_policies(path) -> dict:

    with open(path, mode="r", encoding="utf-8") as fp:
        return json.load(fp)


def process_contract_annotations(contract_name: str,
                                 full_contracts_path: str,
                                 chunk_annots_path: str,
                                 policies_path: str,
                                 output_path: str) -> None:
    # if contract name is not ending with .txt remove the extension and replace it with 
    # .txt to match the naming format of the files containing the full contracts.
    file_name, ext = os.path.splitext(contract_name)
    if ext != ".txt":
        ext = ".txt"
    full_contract_name = f"{file_name}{ext}"
    contract_bin = _data_provider.get_object(full_contracts_path, full_contract_name)
    if not contract_bin:
        logger.info(f"Contract '{full_contract_name}' not found.")
        return
    contract_text = contract_bin.decode()
    contract_item = load_chunk_annots(chunk_annots_path, contract_name)
    contract_chunks = split_contract(contract_text)
    policies = load_policies(policies_path)
    qas_entries = process_cuad_paragraphs(contract_item, contract_chunks, policies)
    output_text = json.dumps({"qas": qas_entries})
    _data_provider.put_object(output_path, contract_name, output_text)


def get_data_provider(chunks_annots_path: str) -> DataProvider:
   
   return S3DataProvider() if chunks_annots_path.startswith("s3://") else LocalDataProvider()


def _init_worker(chunks_annots_path: str):
    
    global _data_provider
    _data_provider = get_data_provider(chunks_annots_path)


def process_chunk_annots(chunk_annots_path: str, 
                         full_contracts_path: str,
                         policies_path: str,
                         output_path: str,
                         max_cpu_count: int=None) -> None:
    
    if max_cpu_count is None:
        max_cpu_count = max(os.cpu_count()-1, 1)

    data_provider_obj = get_data_provider(chunk_annots_path)
    cuad_chunk_names = data_provider_obj.list_objects(chunk_annots_path)
    processed_cuad_chunk_names = data_provider_obj.list_objects(output_path)
    not_processed_cuad_chunk_names = list(set(cuad_chunk_names) - set(processed_cuad_chunk_names))
    
    """
    global _data_provider
    _data_provider = get_data_provider(chunk_annots_path)
    for chunk_name in not_processed_cuad_chunk_names:
        process_contract_annotations(chunk_name, 
                                     full_contracts_path, 
                                     chunk_annots_path, 
                                     policies_path, 
                                     output_path)
    """
    with Pool(processes=max_cpu_count, 
              initializer=_init_worker, 
              initargs=(chunk_annots_path,)) as pool:
        aresults = [pool.apply_async(process_contract_annotations, 
                                     args=(chunk_name, full_contracts_path, chunk_annots_path, 
                                           policies_path, output_path)) 
                    for chunk_name in not_processed_cuad_chunk_names]
        _ = [ar.get() for ar in aresults]
    

if __name__ == "__main__":
    args = parse_args()
    
    process_chunk_annots(args.chunks_annots_path, 
                         args.full_contracts_path,
                         args.policies_path,
                         args.output_path)
