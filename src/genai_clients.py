import json
import http.client
from openai import (
    OpenAI,
    APIStatusError, 
    APIConnectionError, 
    AuthenticationError, 
    RateLimitError
)
from typing import Tuple
from genai_exceptions import *
from abc import ABC, abstractmethod
from openai.types.chat.chat_completion import ChatCompletion
from genai_config import GEN_AI_PROVIDERS as _GEN_AI_PROVIDERS


class GenAIChatMetrics:
    """
    A class to hold chat completion response metrics.
    """
    def __init__(self, 
                 prompt_tokens: int = 0, 
                 completion_tokens: int = 0, 
                 total_tokens: int = 0,
                 cached_tokens: int = 0,
                 prompt_cache_hit_tokens: int = 0,
                 prompt_cache_miss_tokens: int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        self.cached_tokens = cached_tokens
        self.prompt_cache_hit_tokens = prompt_cache_hit_tokens
        self.prompt_cache_miss_tokens = prompt_cache_miss_tokens


class GenAIClient(ABC):
    """ 
    Base class for GenAI clients.
    """
    def __init__(self, **kwargs):
        self._api_parameters = kwargs["api_parameters"]
        self._chat_model = kwargs["chat_model"]
        self._chat_metrics = GenAIChatMetrics()


    def get_chat_metrics(self) -> GenAIChatMetrics:
        return self._chat_metrics


    @abstractmethod
    def _update_metrics(self, response) -> None:
        """
        Updates the chat metrics based on the response from the API.

        Args:
            response: the response object from the API request.
        """
        pass


    @abstractmethod
    def send_request(self, messages: list[dict]) -> str:
        """
        Sends a chat completion request to the OpenAI API and returns the response content.

        Args:
            messages: a list of messages to send to the chat model.

        Returns:
            the content of the response message.
        """
        pass
    

    @abstractmethod
    def get_user_balance(self) -> Tuple[float, str]:
        """
        Returns the user balance and the currency code.

        Returns:
            the user's current balance and the current code.
        """
        pass


    @abstractmethod
    def get_tokens_count(self, text: str) -> int:
        """
        Returns the number of tokens in the given text.

        Args:
            text: the text to count tokens for.

        Returns:
            the number of tokens.
        """
        pass


class OpenAICompatibleClient(GenAIClient):
    """
    A client for OpenAI's compatible APIs. The class implements common functionality
    shared across different GenAI providers that have API compatible to the OpenAI's 
    API, such as DeepSeek.
    """

    def __init__(self, **kwargs):
        """
        Initializes the OpenAI client with the provided API parameters and chat model.
        
        Args:
            kwargs: a dictionary with the following keys:
                - api_parameters: a dictionary with the API parameters, e.g. api_key, base_url, etc.
                - chat_model: a name of the chat model to use, e.g. "gpt-3.5-turbo", "gpt-4", etc.
        """
        super().__init__(**kwargs)
        self.client = OpenAI(**kwargs["api_parameters"])
        

    def _update_metrics(self, response: ChatCompletion) -> None:

        if response.usage is None:
            return
        
        self._chat_metrics.prompt_tokens += response.usage.prompt_tokens
        self._chat_metrics.completion_tokens += response.usage.completion_tokens
        self._chat_metrics.total_tokens += response.usage.total_tokens
        if "prompt_cache_hit_tokens" in response.usage.model_fields:
            self._chat_metrics.prompt_cache_hit_tokens += response.usage.prompt_cache_hit_tokens
        if "prompt_cache_miss_tokens" in response.usage.model_fields:
            self._chat_metrics.prompt_cache_miss_tokens += response.usage.prompt_cache_miss_tokens   
        if response.usage.prompt_tokens_details and response.usage.prompt_tokens_details.cached_tokens:
            self._chat_metrics.cached_tokens += response.usage.prompt_tokens_details.cached_tokens


    def send_request(self, messages: list[dict]) -> str:
        try:
            response = self.client.chat.completions.create(model=self._chat_model,
                                                           messages=messages)
            content = response.choices[0].message.content
            self._update_metrics(response)

            return content
        except RateLimitError as e:
            raise GenAIRateLimitError(f"Rate limit exceeded: {e}") from e
        except AuthenticationError as e:
            raise GenAIAuthError(f"Authentication failed: {e}") from e
        except APIConnectionError as e:
            raise GenAIConnectionError(f"Network error: {e}") from e
        except APIStatusError as e:
            if e.status_code == 402:
                raise GenAIInsufficientBalanceError(f"Insufficient balance: {e}") from e
            if e.status_code == 429:
                raise GenAIRateLimitError(f"Rate limit exceeded: {e}") from e
            if 400 <= e.status_code < 500:
                if "model's maximum context length" in e.message:
                    raise GenAITokenLimitError(f"Token limit exceeded: {e}") from e
                raise GenAIClientError(f"Client error ({e.status_code}): {e}") from e
            elif 500 <= e.status_code < 600:
                raise GenAIServerError(f"Server error ({e.status_code}): {e}") from e
            else:
                raise GenAIUnknownError(f"Unexpected API error: {e}") from e
        except Exception as e:
            raise GenAIUnknownError(f"Unexpected error in OpenAIClient: {e}") from e
        

class DeekSeekAIClient(OpenAICompatibleClient):
    """
    A client for the DeekSeek API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._tokenizer = None


    def get_tokens_count(self, text) -> int:
        return -1
    
    
    def get_user_balance(self) ->  Tuple[float, str]:
        """
        Returns current user balance. Works only for Deep Seek API

        Returns:
            a tuple with the current user balance and the currency short form.
        """
        conn = http.client.HTTPSConnection(self._api_parameters["base_url"].replace("https://", ""))
        headers = {
            "Authorization": f"Bearer {self._api_parameters['api_key']}",
        }
        conn.request("GET", "/user/balance", body="",  headers=headers)
        res = conn.getresponse()
        data = json.loads(res.read().decode("utf-8"))
        currency = data["balance_infos"][0]["currency"]
        balance = float(data["balance_infos"][0]["total_balance"])

        return balance, currency
    

class BedrockAIClient(GenAIClient):

    def __init__(self, **kwargs):
        pass

    
    def send_request(self, messages: list[dict]) -> str:
        return ""


    def get_user_balance(self) -> Tuple[float, str]:
        return (-1, "")
    

    def _update_metrics(self, response) -> None:
        pass
    
    
    def get_tokens_count(self, text) -> int:
        return -1


class GenAIClientFactory:
    
    @staticmethod
    def create(genai_provider_name: str) -> GenAIClient:
        """
        Creates a GenAI client based on the specified provider name.

        Args:
            genai_provider_name: a name of the GenAI provider, e.g. "openai", "bedrockai", etc.

        Returns:
            an instance of a GenAIClient subclass.
        """
        if genai_provider_name in _GEN_AI_PROVIDERS:
            if genai_provider_name == "deep_seek":
                return DeekSeekAIClient(**_GEN_AI_PROVIDERS[genai_provider_name])
            if genai_provider_name == "bedrockai":
                return BedrockAIClient()
            
        raise ValueError(f"Unknown GenAI provider name: {genai_provider_name}")

