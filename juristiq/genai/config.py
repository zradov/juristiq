import os
from types import MappingProxyType 
from juristiq.config.common import DATA_DIR


# DeepSeek API connection settings.
_DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")
_DEEP_SEEK_API_DOMAIN = "api.deepseek.com"
DEEP_SEEK_CHAT_TOKENIZER_DIR = DATA_DIR / "deepseek_v3_tokenizer"


# client options for different GenAI providers.
GEN_AI_PROVIDERS = MappingProxyType(
    dict(
        deep_seek = dict(
            api_parameters = dict(
                api_key=_DEEP_SEEK_API_KEY, 
                base_url=f"https://{_DEEP_SEEK_API_DOMAIN}"
            ),
            chat_model = "deepseek-chat",
            # context length + max output
            max_tokens_count = 136000
        )
    )
)

