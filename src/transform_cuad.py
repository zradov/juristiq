import re
import os
import json
import config
import logging
import argparse
from pathlib import Path
from dotenv import load_dotenv
from multiprocessing import Pool
from logging_config import configure_logging
from data_providers import get_data_provider
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker



configure_logging()
logger = logging.getLogger(__name__)

# Pattern to identify clause type in the "question" field of the CUAD annotation entry.
CLAUSE_TYPE_PATTERN = ".*related to \"(?P<clause_type>[^\"]+)\".*"

_data_provider = None


def parse_args() -> argparse.Namespace:
    """
    Parses and returns CLI options.

    Returns:
        argparse.Namespace object containing the CLI options and values.
    """
    parser = argparse.ArgumentParser(description="CUAD Utilities")
    parser.add_argument("-c", "--chunks_annots_path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket containing chunked CUAD annotations.")
    parser.add_argument("-f", "--full_contracts_path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket containing full contracts directory.")
    parser.add_argument("-p", "--policies-path", 
                        type=str, 
                        required=True, 
                        help="Path to the file containing law documents compliance policies.")
    parser.add_argument("-o", "--output_path", 
                        type=str, 
                        required=True, 
                        help="Path to a local folder or to the S3 bucket where transformed CUAD annotations will be stored.")
    parser.add_argument("-m", "--max-processes", 
                        type=int, 
                        required=False, 
                        default=os.cpu_count()-1,
                        help="The maximum number of processes that will be allowed to run in parallel.")
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
        start1, _ = find_chunk_limits(contract_text, chunk1, offset)
        if start1 != -1:
            start = start1
        else:
            start2, _ = find_chunk_limits(contract_text, chunk2, offset)
            if start2 != -1:
                start = start2 - len(chunk1)
            else:
                return -1, -1
    end = start + len(chunk)

    return start, end


def split_contract(contract_text: str) -> list[dict]:
    """
    Splits constract text into separate chunks based on their semmantic similarity.

    Args:
        contract_text: full text of the contract.

    Returns:
        a list of dict objects containing approximate position of the starting
        and the ending characters of the chunk in the original text and the chunk
        text.
    """
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


def get_clause_type(text: str) -> str:
    """
    Extracts the clause type.

    Args:
        text: a text containing clause type.

    Returns:
        clause type. 
    """
    match = re.match(CLAUSE_TYPE_PATTERN, text)

    if match:
        return match.group("clause_type")
    
    return None


def map_answers_to_chunks(answers: list[dict], 
                          contract_chunks: list[dict]) -> list[str]:
    """
    Maps the answers in the original CUAD annotations, based on their starting character
    position in contracts, to corresponding text chunks. The answers list relates to
    a single annotation entry in the original CUAD annotations.

    Args:
        answers: a list answers in the CUAD annotation entry.
        contract_chunks: chunks of text from a contract.

    Returns:
        a list of text chunks that are related to the to provided answers.
    """
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
    """
    Augments the CUAD annotations entries the additional fields.

    Args:
        questions: a list of questions from the original CUAD dataset.
        contract_chunks: a list of text chunks for the specific contract.
        policies: a list of policies that contract clause need to comply to.

    Returns:
        a list of augmented annotations entries from the original CUAD annotations.
    """
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
            "suggested_redline": "TBD",
            "rationale": "TBD"
        })

    return processed_questions


def process_cuad_paragraphs(data_item: dict, 
                            contract_chunks: list[dict],
                            policies: dict) -> list[dict]:
    """
    Process paragraph entries related to the provided contract's text chunks.

    Args:
        data_item: single annotation entry in the "data" list of the original CUAD annotations.
        contract_chunks: contract's text chunks.
        policies: all policies entries.

    Returns:
        a list of augmented QnA entries related to the original CUAD's "qas" annotations.
    """
    all_entries = []

    for paragraph in data_item["paragraphs"]:
        questions_entries = process_cuad_questions(paragraph["qas"], contract_chunks, policies)
        if questions_entries:
            all_entries.extend(questions_entries)

    return all_entries


def load_chunk_annots(chunk_annots_path: str, 
                      contract_name: str) -> dict:
    """
    Loads the CUAD's chunked annotations groupped by the contract's title.

    Args:
        chunk_annots_path: a path to the folder or S3 bucket where chunked annotations are stored.
        contract_name: the name of the file with chunked contract's annotations.

    Returns:
        a json document with deserialized contract text chunks.
    """
    contract_annots = _data_provider.get_object(chunk_annots_path, contract_name)
    
    return json.loads(contract_annots)
    

def process_contract_annotations(contract_name: str,
                                 full_contracts_path: str,
                                 chunk_annots_path: str,
                                 policies_path: str,
                                 output_path: str) -> None:
    """
    Augments chunked contract annotations with additional fields and 
    stores the augmented chunks in the target location.

    Args:
        contract_name: a name of the contract include the file extension.
        full_contracts_path: a location where the files containing full contracts' text are stored.
        chunk_annots_path: a location where chunked annotations are stored.
        policies_path: a path to the file containing compliance policies.
        output_path: a location where augmented annotations will be stored.

    Returns:
        None
    """
    # if contract name is not ending with .txt remove the extension and replace it with 
    # .txt to match the naming format of the files containing the full contracts.
    file_name, ext = os.path.splitext(contract_name)
    if ext != ".txt":
        ext = ".txt"
    full_contract_name = f"{file_name}{ext}"
    contract_bin = _data_provider.get_object(os.path.join(full_contracts_path, full_contract_name))
    if not contract_bin:
        logger.info(f"Contract '{full_contract_name}' not found.")
        return
    contract_text = contract_bin.decode()
    contract_item = load_chunk_annots(chunk_annots_path, contract_name)
    contract_chunks = split_contract(contract_text)
    policies = json.loads(Path(policies_path).read_text(encoding="utf-8"))
    qas_entries = process_cuad_paragraphs(contract_item, contract_chunks, policies)
    output_text = json.dumps({"qas": qas_entries})
    _data_provider.put_object(os.path.join(output_path, contract_name), output_text)


def _init_worker(chunks_annots_path: str) -> None:
    """
    The initialization function that each worker process will call when it starts.

    Args:
        chunks_annots_path: a location where the files with contracts' chunked annotations are stored.

    Returns:
        None
    """
    global _data_provider
    _data_provider = get_data_provider(chunks_annots_path)


def process_chunk_annots(chunk_annots_path: str, 
                         full_contracts_path: str,
                         policies_path: str,
                         output_path: str,
                         max_processes: int=None) -> None:
    """
    Runs the CUAD annotations augmentation process for each contract in the CUAD dataset.

    Args:
        chunks_annots_path: a location where the files with contracts' chunked annotations are stored.
        full_contracts_path: a location where the files containing full contracts' text are stored.
        policies_path: a path to the file containing compliance policies.
        output_path: a location where augmented annotations will be stored.
        max_cpu_count: the maximum number of parallel processes.
    """
    if max_processes is None:
        max_processes = max(os.cpu_count()-1, 1)

    data_provider_obj = get_data_provider(chunk_annots_path)
    cuad_chunk_names = data_provider_obj.list_objects(chunk_annots_path)
    processed_cuad_chunk_names = data_provider_obj.list_objects(output_path)
    not_processed_cuad_chunk_names = list(set(cuad_chunk_names) - set(processed_cuad_chunk_names))
    
    with Pool(processes=max_processes, 
              initializer=_init_worker, 
              initargs=(chunk_annots_path,)) as pool:
        aresults = [pool.apply_async(process_contract_annotations, 
                                     args=(chunk_name, full_contracts_path, chunk_annots_path, 
                                           policies_path, output_path)) 
                    for chunk_name in not_processed_cuad_chunk_names]
        _ = [ar.get() for ar in aresults]
    

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    
    process_chunk_annots(args.chunks_annots_path, 
                         args.full_contracts_path,
                         args.policies_path,
                         args.output_path,
                         args.max_processes)
