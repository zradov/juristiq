import hashlib


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
