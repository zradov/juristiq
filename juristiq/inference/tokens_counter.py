import json
from typing import Dict
from abc import ABC, abstractmethod
from botocore.client import BaseClient
from transformers import GPT2TokenizerFast
from juristiq.inference.models import ModelName
from juristiq.cloud.utils import get_bedrock_runtime_client
from juristiq.inference.prompts import get_prompt_text
from juristiq.config.consts import TOKENS_RATIO_FILE_PATH


class TokensCounter(ABC):
    """Abstract base class for counting tokens in prompts."""

    def __init__(self, tokenizer, model_name: ModelName):
        self._tokenizer = tokenizer
        self._model_name = model_name
        if TOKENS_RATIO_FILE_PATH.exists():
            self._tokens_ratio = json.loads(TOKENS_RATIO_FILE_PATH.read_text(encoding="utf8"))
        else:
            self._tokens_ratio = {"tokenizers": {}}


    @abstractmethod
    def count_tokens(self, prompt: Dict) -> int:
        pass


    @classmethod
    def create(cls, model_name: ModelName) -> "TokensCounter":
        
        if model_name == ModelName.NOVA_LITE:
            return LocalTokensCounter(GPT2TokenizerFast.from_pretrained("gpt2"), model_name)
        
        raise ValueError(f"Unsupported model name: {model_name}.")


class LocalTokensCounter(TokensCounter):
    """
    Tokens counter that uses a tokenizer on a workstation that is 
    running the token's counter script.
    """

    def count_tokens(self, prompt: str | Dict) -> int:

        prompt_text = prompt if isinstance(prompt, str) else get_prompt_text(prompt) 

        tokens = self._tokenizer.encode(prompt_text, 
                                        add_special_tokens=False, 
                                        truncation=False)

        return len(tokens)


class RemoteTokensCounter(TokensCounter):
    """Tokens counter that uses AWS Bedrock's count_tokens API to count tokens."""
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        self._client = None


    def count_tokens(self, prompt: Dict) -> int:

        if not self._client:
            self._client = get_bedrock_runtime_client()

        res = self._client.count_tokens(modelId=self._model_name.value, 
                                        input={"converse": prompt})
        
        tokens_count = res["modelOutput"]["inputTextTokenCount"]
        tokens_coef = self._tokens_ratio.get(self._model_name, 1)
        tokens_count = round(tokens_count / tokens_coef)

        return tokens_count
