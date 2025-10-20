import json
from typing import List, Tuple
from pydantic import BaseModel, Field, ConfigDict


_model_config = ConfigDict(extra="forbid", validate_assignment=True, frozen=True)


class InferenceParams(BaseModel):
    """ Base class for inference parameters """

    temperature: float = Field(ge=0.0, le=2.0, default=0.0, description="Temperature for sampling")
    top_p: float = Field(ge=0.0, le=1.0, default=0.99, description="Top-p for nucleus sampling")
    max_tokens: int = Field(ge=1, le=512, default=256, description="Maximum tokens to generate")
    stop_sequences: Tuple[str, ...] = Field(default_factory=tuple)

    model_config = _model_config


class BatchInferenceParams(InferenceParams):
    """Inference parameters for Bedrock's batch inference processing"""

    top_k: int = Field(ge=1, le=40, default=40)

    def __str__(self):
        return self.model_dump_json()


class JudgeInferenceParams(InferenceParams):
    """Inference parameters for Models-as-Judge evaluations"""

    def __str__(self):
        data = {}
        fields = self.__class__.model_fields

        for key in fields:
            parts = key.split("_")
            new_key = parts[0] + "".join([s.capitalize() for s in parts[1:]])
            data[new_key] = getattr(self, key)

        return json.dumps(data)
    

DEFAULT_BATCH_INFERENCE_PARAMS = BatchInferenceParams()
DEFAULT_JUDGE_INFERENCE_PARAMS = JudgeInferenceParams()
CUSTOM_EVALUATION_METRIC_NAME = "BedrockCustomEvaluationMetric"

