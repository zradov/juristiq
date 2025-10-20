from enum import Enum
from typing import NamedTuple
from juristiq.inference.models import ModelName


class TokenCosts(NamedTuple):
    """Cost per input and output tokens."""
    input: float
    output: float


class ProcessingMode(Enum):
    """Types of prompt data processing"""
    ON_DEMAND = 0
    BATCH = 1


class ProcessingCosts(NamedTuple):
    batch: TokenCosts
    on_demand: TokenCosts

    def __getitem__(self, mode: ProcessingMode) -> TokenCosts:

        if mode == ProcessingMode.ON_DEMAND:
            return self.on_demand
        elif mode == ProcessingMode.BATCH:
            return self.batch
        
        raise KeyError(f"Unknown processing mode: {mode}")


class TokensMetrics(NamedTuple):
    """Metrics for total input and output tokens."""
    total_input_tokens: int
    total_output_tokens: int


"""Costs per 1000 tokens for different models."""
_COSTS_PER_1000_TOKENS = {
    ModelName.NOVA_LITE: ProcessingCosts(
        on_demand=TokenCosts(input=0.00006, output=0.00024),
        batch=TokenCosts(input=0.00003, output=0.00012)
    )
}


def calculate_costs(model: ModelName, 
                    tokens_metrics: TokensMetrics,
                    processing_mode: ProcessingMode=ProcessingMode.ON_DEMAND) -> TokenCosts:
    """
    Calculate the costs based on the model and token metrics.
    
    Args:
        model: The model name.
        processing_mode: The mode of processing.
        tokens_metrics: The token metrics containing total input and output tokens.

    Returns:
        A TokensCost named tuple with calculated costs for input and output tokens.
    """
    model_costs = _COSTS_PER_1000_TOKENS[model][processing_mode]
    tokens_cost = TokenCosts(input=model_costs.input * (tokens_metrics.total_input_tokens // 1000),
                             output=model_costs.output * (tokens_metrics.total_output_tokens // 1000))

    return tokens_cost
