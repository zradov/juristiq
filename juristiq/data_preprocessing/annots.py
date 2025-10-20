import json
import hashlib
from typing import List, Dict


def get_hash(messages: list[dict]) -> str:
    """
    Returns a hash value for a string containing the provided messages.
    Used primarely to calculate the hash value of system and user messages
    for LLM prompt.

    Args:
        messages: a list of message dictionaries.

    Returns:
        a hash value.
    """
    messages_text = "\n".join([f"{k}:{v}" 
                               for message in messages 
                               for k,v in message.items() ])
    res = hashlib.md5(messages_text.encode())
    
    return res.hexdigest()


def load_data_from_jsonl(file_path: str) -> List[Dict]:
    """
    Loads records from .jsonl file.

    Args:
        file_path: a local file system path to the .jsonl file.

    Returns:
        a list of records.
    """
    records = []

    with open(file_path, mode="r", encoding="utf8") as fp:
        for line in fp:
            records.append(json.loads(line))

    return records
