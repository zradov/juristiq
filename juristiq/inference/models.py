from enum import Enum


class ModelName(Enum):
    """Enumeration of supported model names."""
    NOVA_LITE = "amazon.nova-lite-v1:0"
    NOVA_PRO = "amazon.nova-pro-v1:0"
    GPT_OSS_20B = "openai.gpt-oss-20b-1:0"

