import json
from enum import Enum
from aws_cdk import Aws
from aws_cdk import (
    Aws,
    Stack,
    aws_iam as iam
)
from pathlib import Path
from string import Template
from iac_config import S3_BUCKET_NAME_PREFIX


class PermissionsType(Enum):

    BEDROCK_BATCH_INFERENCE = 0


def _get_bedrock_batch_inference_parameters(**kwargs) -> dict:

    return {
        "model_name": kwargs.get("model_name", ""),
        "account_id": Aws.ACCOUNT_ID,
        "region_name": Aws.REGION,
        "s3_input_bucket": f"{S3_BUCKET_NAME_PREFIX}-{kwargs.get('s3_input_bucket', 'batch-inference-input')}",
        "s3_output_bucket": f"{S3_BUCKET_NAME_PREFIX}-{kwargs.get('s3_output_bucket', 'batch-inference-output')}",
        "bedrock_service_role": kwargs.get("bedrock_service_role", "")
    }


def _get_parameters(permissions_type: PermissionsType, **kwargs):
    
    if permissions_type ==  PermissionsType.BEDROCK_BATCH_INFERENCE:
        return _get_bedrock_batch_inference_parameters(**kwargs) 
    
    raise ValueError(f"Permissions type '{permissions_type} is not valid.")


def get_policy_doc(policy_template_path: str, permissions_type: PermissionsType, **kwargs) -> dict:

    policy_template_text = Path(policy_template_path).read_text(encoding="utf8")
    policy_template = Template(policy_template_text)
    parameters = _get_parameters(permissions_type, **kwargs)
    policy_text = policy_template.substitute(**parameters)
    policy_json = json.loads(policy_text)

    return policy_json


def attach_policy_to_role(stack: Stack,
                          policy_name: str,
                          policy_template_path: str, 
                          permissions_type: PermissionsType, 
                          role: iam.Role,
                          **kwargs) -> None:
    
    policy_text = get_policy_doc(policy_template_path, permissions_type, **kwargs)
    policy_doc = iam.PolicyDocument.from_json(policy_text)
    policy = iam.ManagedPolicy(stack, 
                               policy_name, 
                               document=policy_doc, 
                               managed_policy_name=policy_name)
    role.add_managed_policy(policy)
