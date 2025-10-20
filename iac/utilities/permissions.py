import json
import iac_config
from enum import Enum
from aws_cdk import Aws
from aws_cdk import (
    Aws,
    Stack,
    aws_iam as iam
)
from string import Template
from typing import List, Dict


class PermissionsType(Enum):
    """
    Enum for different types of permissions.
    """
    BEDROCK_BATCH_INFERENCE = 0
    BEDROCK_EVALUATION = 1


_POLICIES_TEMPLATE_PATHS = {
    PermissionsType.BEDROCK_BATCH_INFERENCE: iac_config.BEDROCK_BATCH_INFERENCE_SUBMITTER_TEMPLATE_PATH,
    PermissionsType.BEDROCK_EVALUATION: iac_config.BEDROCK_MODEL_AS_JUDGE_EVALUATION_TEMPLATE_PATH
}


def _get_base_params(**kwargs) -> Dict:
    """
    Get base parameters for policy templates.

    Returns:
        dict: Base parameters including account ID and region name.
    """
    return {
        "account_id": Aws.ACCOUNT_ID,
        "region_name": Aws.REGION
    }


def _get_bedrock_batch_inference_params(**kwargs) -> Dict:
    """
    Get parameters for Bedrock Batch Inference policy template.

    Returns:
        dict: Parameters specific to Bedrock Batch Inference.
    """
    return {
        **_get_base_params(**kwargs),
        "model_name": kwargs.get("model_name", ""),
        "bedrock_service_role": kwargs.get("bedrock_service_role", ""),
        "s3_input_bucket": f"{iac_config.S3_BUCKET_NAME_PREFIX}-{kwargs.get("s3_input_bucket", iac_config.BEDROCK_BATCH_INFERENCE_INPUT_BUCKET_NAME)}",
        "s3_output_bucket": f"{iac_config.S3_BUCKET_NAME_PREFIX}-{kwargs.get("s3_output_bucket", iac_config.BEDROCK_BATCH_INFERENCE_OUTPUT_BUCKET_NAME)}",   
    }


def _get_bedrock_evaluation_params(**kwargs) -> Dict:
    """
    Get parameters for Bedrock Evaluation policy template.

    Returns:
        dict: Parameters specific to Bedrock Evaluation.
    """
    return {
        **_get_base_params(**kwargs),
        "s3_input_bucket": f"{iac_config.S3_BUCKET_NAME_PREFIX}-{kwargs.get("s3_input_bucket", iac_config.BEDROCK_EVALUATION_BUCKET_NAME)}",
        "s3_results_folder": f"{iac_config.S3_BUCKET_NAME_PREFIX}-{kwargs.get("s3_results_folder", iac_config.BEDROCK_EVALUATION_RESULTS_FOLDER)}",        
    }


def _get_parameters(permissions_type: PermissionsType, **kwargs) -> Dict:
    """
    Get parameters for the specified permissions type.

    Args:
        permissions_type: the type of permissions

    Returns:
        parameters dictionary for the policy template.
    """
    if permissions_type == PermissionsType.BEDROCK_BATCH_INFERENCE:
        return _get_bedrock_batch_inference_params(**kwargs) 
    if permissions_type == PermissionsType.BEDROCK_EVALUATION:
        return _get_bedrock_evaluation_params(**kwargs)
    
    raise ValueError(f"Permissions type '{permissions_type} is not valid.")


def _get_policy_doc(permissions_type: PermissionsType, **kwargs) -> Dict:
    """
    Get the policy document for the specified permissions type.

    Args:
        permissions_type: the type of permissions.

    Returns:
        policy document as a dictionary.
    """

    policy_template_text = _POLICIES_TEMPLATE_PATHS[permissions_type].read_text(encoding="utf8")
    policy_template = Template(policy_template_text)
    parameters = _get_parameters(permissions_type, **kwargs)
    policy_text = policy_template.substitute(**parameters)
    policy_json = json.loads(policy_text)

    return policy_json


def get_service_principal(service_name: str,
                          arn_equals_conditions: Dict = None) -> iam.ServicePrincipal:
    """
    Get a service principal with optional ARN equals conditions.

    Args:
        service_name: the name of the AWS service.
        arn_equals_conditions: optional dictionary of ARN equals conditions.

    Returns:
        the service principal.
    """

    conditions = {"StringEquals": {"aws:SourceAccount": Aws.ACCOUNT_ID }}

    if arn_equals_conditions:
        conditions["ArnEquals"] = arn_equals_conditions

    principal = iam.ServicePrincipal(
        service_name,
        conditions=conditions
    )

    return principal


def get_policy(stack: Stack,
               policy_name: str,
               permissions_type: PermissionsType,
               **kwargs) -> iam.ManagedPolicy:
    """
    Get a managed policy for the specified permissions type.

    Args:
        stack: the CDK stack.
        policy_name: the name of the policy.
        permissions_type: the type of permissions.

    Returns:
        the managed policy.
    """

    policy_text = _get_policy_doc(permissions_type, **kwargs)
    policy_doc = iam.PolicyDocument.from_json(policy_text)
    policy = iam.ManagedPolicy(stack, 
                               policy_name, 
                               document=policy_doc, 
                               managed_policy_name=policy_name)

    return policy


def create_role(stack: Stack,
                role_name: str,
                principal: iam.PrincipalBase,
                description: str,
                policies: List[iam.ManagedPolicy] | iam.ManagedPolicy = None) -> iam.Role:
    """
    Create an IAM role with the specified parameters.

    Args:
        stack: the CDK stack.
        role_name: the name of the role.
        principal: the principal that can assume the role.
        description: the description of the role.
        policies: optional list of managed policies to attach to the role.

    Returns:
        the created IAM role.
    """
    role = iam.Role(stack, role_name, assumed_by=principal, description=description)

    if policies:

        if not isinstance(policies, list):
            policies = [policies]

        for policy in policies:
            role.add_managed_policy(policy)

    return role
