import yaml
import config
from genai_clients import (
    GenAIClient,
    OpenAIClient
)
from pathlib import Path
from jinja2 import Template
from typing import Callable, Dict, Any, Optional
from genai_config import GEN_AI_PROVIDERS as _GEN_AI_PROVIDERS


def _filter(text: str) -> str:
    return text.replace("\n", r"\\n")


def get_genai_client(genai_provider_name: str) -> GenAIClient:
    """
    Creates and returns an OpenAI client object.

    Args:
        genai_provider_name: a name of the GenAI provider.

    Returns:
        a GenAIClient object or raises a KeyError in case when the GenAI provider is not supported.
    """
    if genai_provider_name in _GEN_AI_PROVIDERS:
        if genai_provider_name in ["deep_seek", "open_ai"]:
            return OpenAIClient(**_GEN_AI_PROVIDERS[genai_provider_name]) 
    
    raise KeyError(f"Unknown GenAI provider name: {genai_provider_name}")


def _get_prompt_template(template_path: str) -> str:
    """
    Loads and returns the prompt template contents

    Args:
        template_path: a local file system path to the prompt template.

    Returns:
        a prompt template string.
    """
    path = Path(template_path)
    prompt_template = path.read_text(encoding="utf-8")

    return prompt_template


def _create_prompt_builder(
    template_path: str,
    variable_mapping: Optional[Dict[str, Callable[[Dict, Any], str]]] = None
) -> Callable[..., Dict]:
    """
    Returns a function that creates prompt messages from a template.
    
    Args:
        template_path: a path to the prompt template file
        variable_mapping: a dictionary mapping template variable names to functions 
                          that extract the value from input data. If None, uses 
                          default contract review mapping.
    
    Returns:
        a function that builds prompts using the specified template and mapping.
    """
    prompt_template = _get_prompt_template(template_path)
    jinja_template = Template(prompt_template)

    def builder(**kwargs: Any) -> list[Dict]:
        """
        Builds a prompt by formatting the template with data.
        
        Args:
            kwargs: additional named context parameters
        
        Returns:
            Dictionary containing the formatted prompt messages
        """
        # Build template variables by applying mapping functions
        template_vars = {}
        for var_name, extractor in variable_mapping.items():
            template_vars[var_name] = extractor(kwargs)
        
        temp_prompt_template = prompt_template
        if "context" in template_vars:
            temp_prompt_template = jinja_template.render(template_vars["context"])
            del template_vars["context"]
        formatted_prompt = temp_prompt_template.format(**template_vars)
        messages_dict = yaml.safe_load(formatted_prompt)
        messages = [{"role": k, **v} for k, v in messages_dict.items()]
        
        return messages
        
    return builder


def _get_prompt_builder(
    template_path: str, 
    context_transformer: Callable,
    context_key: str = "data_batch"
) -> Callable[..., list[Dict]]:
    """
    Returns a prompt builder function with the specified configuration.
    
    Args:
        template_path: Path to the prompt template
        context_transformer: Function to transform the data_batch into context
        context_key: Key name for the data batch in kwargs (default: "data_batch")
    """
    prompt_build_vars = {
        "versions_count": lambda kwargs: kwargs["versions_count"],
        "context": lambda kwargs: context_transformer(kwargs[context_key])
    }
    
    return _create_prompt_builder(template_path, prompt_build_vars)


def get_clause_prompt_builder() -> Callable[..., list[Dict]]:
    """
    Returns a function that creates prompt messages for clause augmentation.

    Returns:
        a function that parses the prompt template for clause augmentation.
    """
    return _get_prompt_builder(
        config.CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH,
        lambda data_batch: {"samples": list(data_batch)}
    )


def get_missing_clause_prompt_builder() -> Callable[..., list[Dict]]:
    """
    Returns a function that creates prompt messages for missing clause augmentation.

    Returns:
        a function that parses the prompt template for missing clause augmentation.
    """
    return _get_prompt_builder(
        config.MISSING_CLAUSE_AUGMENTATION_LLM_PROMPT_TEMPLATE_PATH,
        lambda data_batch: {"clauses": list(data_batch)}
    )


def get_rephrase_text_prompt_builder() -> Callable[..., list[Dict]]:
    """
    Returns a function that creates prompt messages for rephrasing input text.

    Returns:
        a function that parses the prompt template for rephrasing input text.
    """
    return _get_prompt_builder(
        config.REPHRASE_TEXT_LLM_PROMPT_TEMPLATE_PATH,
        lambda data_batch: "\n    ".join(list(data_batch)),
        "titles"
    )


def get_contract_review_prompt_builder() -> Callable[..., list[Dict]]:
    """
    Returns a function that creates prompt messages for contract compliance review.

    Returns:
        a function that parses the prompt template for contract compliance review.
    """
    prompt_build_vars = dict(
        contract_excerpt=lambda kwargs: _filter(kwargs["versions_count"],
        clause_type=lambda kwargs: _filter(kwargs["item"]["context"]),
        policy_id=lambda kwargs: _filter(kwargs["item"]["policy_id"]),
        policy_text=lambda kwargs: _filter(kwargs["policies"][kwargs["item"]["clause_type"]]["policy_text"]))
    )
    prompt_builder = _create_prompt_builder(config.CONTRACT_REVIEW_LLM_PROMPT_TEMPLATE_PAT,
                                            prompt_build_vars)
    
    return prompt_builder
