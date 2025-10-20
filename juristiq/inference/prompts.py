import json
from enum import Enum
from pathlib import Path
import juristiq.config.templates as templates
from typing import Optional, List, Union, Dict
from juristiq.inference.models import ModelName
from pydantic import BaseModel, Field, ConfigDict


class ModelTokenizer(Enum):
    NOVA_LITE = "gpt2"


class ModelType(Enum):
    """Enumeration of supported model types."""
    CLAUDE = "claude"
    TITAN = "titan"
    LLAMA = "llama"
    NOVA = "nova"


# Base configuration
class BaseGenerationConfig(BaseModel):
    """Base configuration for text generation models."""

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
    """Prompt structure for Titan models."""

    inputText: str
    textGenerationConfig: Dict

    def model_dump(self, *args, **kwargs) -> Dict:

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
    """Prompt structure for Llama models."""

    prompt: str
    temperature: float
    top_p: float
    max_gen_len: int

    def model_dump(self, *args, **kwargs) -> Dict:
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
    """Prompt structure for Claude models."""

    anthropic_version: str = "bedrock-2023-05-31"
    # anthropic_beta: List[str] = Field(default_factory=lambda: ["computer-use-2024-10-22"])
    max_tokens: int
    system: Optional[str] = None
    messages: List[Dict]
    temperature: float
    top_p: float
    top_k: int

    def model_dump(self, *args, **kwargs) -> Dict:
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
    """Prompt structure for Nova models."""

    system: Optional[List[NovaSystemMessage]] = None
    messages: List[NovaMessage]
    inferenceConfig: Optional[NovaInferenceConfig] = None

    def model_dump(self, *args, **kwargs) -> Dict:
        base_Dict = super().model_dump(*args, **kwargs)
        # Remove None values
        return {k: v for k, v in base_Dict.items() if v is not None}
    
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
        A Dictionary representing the request body for the specified model type.
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


def get_batch_inference_record(
    text: str,
    record_id: str,
    model_type: ModelType,
    base_config: Optional[BaseGenerationConfig] = None
) -> str:
    """
    Creates and returns a batch inference record.
    
    Args:
        text: Text that will be stored as user's content
        record_id: a record ID
        model_type: Type of model to generate prompts for
        base_config: Default configuration to use for missing values

    Returns:
        A JSONL string representing the record.
    """
    if base_config is None:
        base_config = BaseGenerationConfig()
    
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


def get_prompt_text(prompt: Dict) -> str:
    """
    Create a text representation of the given prompt.

    Args:
        prompt (Dict): A Dictionary containing the prompt details.

    Returns:
        a string representing the prompt.
    """
    return (
        f"{prompt['system'][0]['text']}"
        "\n"
        f"{prompt["messages"][0]["content"][0]["text"]}"
    )


def get_assistant_message(annot: Dict) -> str:
    """
    Create an assistant message for the given annotation.

    Args:
        annot (Dict): A Dictionary containing the annotation details.

    Returns:
        Dict: A Dictionary representing the assistant message.
    """
    return {
        "role": "assistant",
        "content": [
            {
                "text": get_annot_output(annot)
            }
        ]
    }


def get_system_message(system_prompt: str) -> Dict:
    """
    Creates and returns system prompt messages.

    Args:
        system_prompt: a system prompt text

    Returns:
        a Dictionary with the system prompt message properties.
    """
    return {
        "system": [
            { "text": system_prompt },
            { "cachePoint": { "type": "default" }}]
    }


def get_user_message(annot: Dict) -> Dict:
    """
    Create a user message for the given annotation.

    Args:
        annot (Dict): A Dictionary containing the annotation details.

    Returns:
        Dict: A Dictionary representing the user message.
    """
    return {
        "role": "user",
        "content": [
            {
                "text": get_annot_input(annot)
            }
        ]
    }


def get_annot_input(annot: Dict) -> str:
    """
    Create a text representation of the given annotation.

    Args:
        annot (Dict): A Dictionary containing the annotation details.

    Returns:
        str: A string representing the annotation.
    """
    return (f"Question: {annot['question']} Context: {annot['context']} "
            f"Policy: {annot['policy_text']} Clause Type: {annot['clause_type']}")


def get_annot_output(annot: Dict) -> str:
    """
    Returns expected prompt request output for the provided annotation sample.

    Args:
        annot: annotation sample

    Returns:
        an expected output of a prompt request.
    """
    return (
        f"Review label: {annot['review_label']}. "
        f"Rationale: {annot['rationale']} "
        f"Suggested Redline: {annot['suggested_redline']}"
    )


def get_judge_evaluation_prompt(model_name: ModelName, annot: Dict) -> str:
    """
    Create a judge evaluation prompt for the given model and annotation.

    Args:
        model_name: the name of the model.
        annot: a dictionary containing the annotation details.

    Returns:
        a string representing the judge evaluation prompt.
    """
    system_prompt_text = get_system_prompt_text(model_name)
    user_prompt_text = get_annot_input(annot)
    prompt = f"{system_prompt_text}\n### TASK INPUTS\n\n{user_prompt_text}" 

    return prompt


def get_system_prompt_text(model_name: ModelName) -> str:
    """
    Get the system prompt text for the specified model name.

    Args:
        model_name: The name of the model.

    Returns:
        the system prompt text as a string.
    """
    if model_name == ModelName.NOVA_LITE:
        return Path(templates.AMAZON_NOVA_EVALUATION_SYSTEM_ROLE_PROMPT).read_text(encoding="utf8")

    raise ValueError(f"Unsupported model name: {model_name}.")
