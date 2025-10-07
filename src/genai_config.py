import os
from types import MappingProxyType 


# DeepSeek API connection settings.
_DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")
_DEEP_SEEK_API_DOMAIN = "api.deepseek.com"


# client options for different GenAI providers.
GEN_AI_PROVIDERS = MappingProxyType(
    dict(
        deep_seek = dict(
            api_parameters = dict(
                api_key=_DEEP_SEEK_API_KEY, 
                base_url=f"https://{_DEEP_SEEK_API_DOMAIN}"
            ),
            chat_model = "deepseek-chat"
        )
    )
)

