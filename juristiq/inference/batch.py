# This is customized version of the script:
# https://github.com/aws-samples/amazon-bedrock-samples/blob/main/introduction-to-bedrock/batch_api/batchhelper.py 

import json
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Union


class ModelType(Enum):
    """Enumeration of supported model types."""
    CLAUDE = "claude"
    TITAN = "titan"
    LLAMA = "llama"
    NOVA = "nova"

# Base configuration
class BaseGenerationConfig(BaseModel):
    temperature: float = 0.0
    top_p: float = 0.99
    max_tokens: int = 256
    stop_sequences: Optional[List[str]] = Field(default_factory=list)
    top_k: Optional[int] = None
    system: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True
    )


class TitanPrompt(BaseModel):

    inputText: str
    textGenerationConfig: dict


    def model_dump(self, *args, **kwargs) -> dict:

        return {
            "inputText": self.inputText,
            "textGenerationConfig": {
                "temperature": self.textGenerationConfig["temperature"],
                "topP": self.textGenerationConfig["top_p"],
                "maxTokenCount": self.textGenerationConfig["max_tokens"],
                "stopSequences": self.textGenerationConfig["stop_sequences"]
            }
        }
    

    @classmethod
    def from_data(cls, 
                  text: str, 
                  config: BaseGenerationConfig) -> "TitanPrompt":

        return cls(
            inputText=text,
            textGenerationConfig={
                "temperature": config.temperature,
                "top_p": config.top_p,
                "max_tokens": config.max_tokens,
                "stop_sequences": config.stop_sequences
            }
        )


class LlamaPrompt(BaseModel):
    
    prompt: str
    temperature: float
    top_p: float
    max_gen_len: int

    def model_dump(self, *args, **kwargs) -> dict:
        return {
            "prompt": self.prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_gen_len": self.max_gen_len
        }
    

    @classmethod
    def from_data(cls, 
                  prompt: str, 
                  config: BaseGenerationConfig)  -> "LlamaPrompt":
        
        return cls(
            prompt=prompt,
            temperature=config.temperature,
            top_p=config.top_p,
            max_gen_len=config.max_tokens
        )


class ClaudePrompt(BaseModel):

    anthropic_version: str = "bedrock-2023-05-31"
    # anthropic_beta: List[str] = Field(default_factory=lambda: ["computer-use-2024-10-22"])
    max_tokens: int
    system: Optional[str] = None
    messages: List[dict]
    temperature: float
    top_p: float
    top_k: int


    def model_dump(self, *args, **kwargs) -> dict:
        return {
            "anthropic_version": self.anthropic_version,
            # "anthropic_beta": self.anthropic_beta,
            "max_tokens": self.max_tokens,
            "system": self.system,
            "messages": self.messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k
        }
    

    @classmethod
    def from_data(cls, 
                  prompt: str, 
                  config: BaseGenerationConfig)  -> "ClaudePrompt":
        
        return cls(
            max_tokens=config.max_tokens,
            system=config.system,
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k or 50
        )


class NovaTextContent(BaseModel):
    text: str


class NovaMessage(BaseModel):
    role: str = "user"
    content: List[NovaTextContent]


class NovaInferenceConfig(BaseModel):
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stopSequences: Optional[List[str]] = None


class NovaSystemMessage(BaseModel):
    text: str


class NovaPrompt(BaseModel):

    system: Optional[List[NovaSystemMessage]] = None
    messages: List[NovaMessage]
    inferenceConfig: Optional[NovaInferenceConfig] = None


    def model_dump(self, *args, **kwargs) -> dict:
        base_dict = super().model_dump(*args, **kwargs)
        # Remove None values
        return {k: v for k, v in base_dict.items() if v is not None}
    

    @classmethod
    def from_data(cls,
                  text: str,
                  config: BaseGenerationConfig) -> "NovaPrompt":
        
        nova_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_new_tokens": config.max_tokens,
            "top_k": config.top_k,
            "stopSequences": config.stop_sequences
        }
        
        prompt = cls(
            messages=[
                NovaMessage(
                    content=[NovaTextContent(text=text)]
                )
            ],
            inferenceConfig=NovaInferenceConfig(**nova_config)
        )
        
        if config.system:
            prompt.system = [NovaSystemMessage(text=config.system)]

        return prompt
            

def _get_request_body(text: str, 
                      model_type: ModelType, 
                      config: Optional[BaseGenerationConfig] = None) -> \
                        Union[TitanPrompt, LlamaPrompt, ClaudePrompt, NovaPrompt]:
    """
    Generates the appropriate request body based on the model type and configuration.

    Args:
        text: The input text for the model.
        model_type: The type of model to generate the request for.
        config: Optional configuration for the model. If not provided, defaults will be used.

    Returns:
        A dictionary representing the request body for the specified model type.
    """
    
    if config is None:
        config = BaseGenerationConfig()

    if model_type == ModelType.TITAN:
        return TitanPrompt.from_data(text, config).model_dump()
    
    elif model_type == ModelType.LLAMA:
        return LlamaPrompt.from_data(text, config).model_dump()
    
    elif model_type == ModelType.CLAUDE:
        return ClaudePrompt.from_data(text, config).model_dump()
    
    elif model_type == ModelType.NOVA:
        return NovaPrompt.from_data(text, config).model_dump()
    
    raise ValueError(f"Unknown model type: {model_type}")


def _generate_record_id(index: int, prefix: str = "REC") -> str:
    """Generate an 11 character alphanumeric record ID."""
    return f"{prefix}{str(index).zfill(8)}"


def get_batch_record(
    text: str,
    record_index: int,
    model_type: ModelType,
    base_config: Optional[BaseGenerationConfig] = None
) -> str:
    """
    Converts an annotation dictionary to a JSONL file for batch inference.
    
    Args:
        text: Text that will be stored as user's content
        record_index: an index of the record that will be used to generate the record ID
        model_type: Type of model to generate prompts for
        base_config: Default configuration to use for missing values

    Returns:
        A JSONL string representing the record.
    """
    if base_config is None:
        base_config = BaseGenerationConfig()
    
    record_id = _generate_record_id(record_index)
    
    row_config = BaseGenerationConfig(**base_config.model_dump())
    
    record = {
        "recordId": record_id,
        "modelInput": _get_request_body(
            text=text,
            model_type=model_type,
            config=row_config
        )
    }
            
    return json.dumps(record, ensure_ascii=False) + '\n'
