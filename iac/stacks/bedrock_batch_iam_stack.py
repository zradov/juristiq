import iac_config
from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_s3 as s3,
    aws_ssm as ssm,
    Aws,
    CfnOutput
)
from typing import Tuple
from constructs import Construct
from aws_cdk.aws_s3 import IBucket
from utilities.permissions import (
    get_policy,
    PermissionsType,
    get_service_principal,
    create_role
)


def _get_s3_buckets(stack: Stack) -> Tuple[str, str]:

    input_bucket = stack.node.try_get_context("s3_input_bucket") or \
        iac_config.BEDROCK_BATCH_INFERENCE_INPUT_BUCKET_NAME
    output_bucket = stack.node.try_get_context("s3_output_bucket") or \
        iac_config.BEDROCK_BATCH_INFERENCE_OUTPUT_BUCKET_NAME

    return input_bucket, output_bucket


def _create_service_role(stack: Stack,
                         model_name: str,
                         s3_input_bucket: IBucket,
                         s3_output_bucket: IBucket) -> iam.Role:

    principal = get_service_principal("bedrock.amazonaws.com",
                                      arn_equals_conditions={{
                                          "aws:SourceArn": f"arn:aws:bedrock:{Aws.REGION}:{Aws.ACCOUNT_ID}:model-invocation-job/*"}
                                      })
    policy = get_policy(stack,
                          "BedrockBatchInferencePolicy",
                          PermissionsType.BEDROCK_BATCH_INFERENCE,
                          s3_input_bucket=s3_input_bucket,
                          s3_output_bucket=s3_output_bucket,
                          model_name=model_name)
    role = create_role(stack,
                       "BedrockBatchInferenceServiceRole",
                       principal,
                       description="Service role for creating and running batch inference jobs.",
                       policies=policy)
    
    return role


def _create_submitter_role(stack: Stack, 
                           user_name: str, 
                           bedrock_service_role: str) -> iam.Role:

    submitter_policy = get_policy(stack,
                                  "BedrockBatchInferenceSubmitterPolicy",
                                  PermissionsType.BEDROCK_BATCH_INFERENCE,
                                  bedrock_service_role=bedrock_service_role)
    role = create_role(stack,  
                       "BedrockBatchInferenceSubmitterRole",
                       principal=iam.ArnPrincipal(f"arn:aws:iam::{Aws.ACCOUNT_ID}:user/{user_name}"),
                       description="Role used by ML pipeline or developers to submit/manage Bedrock batch inference jobs",
                       policies=submitter_policy)
    
    return role


class BedrockBatchIAMStack(Stack):
    """
    Stack to create IAM roles and policies for Bedrock Batch Inference jobs.
    """
    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        s3_input_bucket, s3_output_bucket = _get_s3_buckets(self)

        service_role = _create_service_role(self,
                                            model_name=self.node.try_get_context("model_name") or \
                                                iac_config.DEFAULT_BEDROCK_BATCH_INFERENCE_MODEL,
                                            s3_input_bucket=s3_input_bucket,
                                            s3_output_bucket=s3_output_bucket)
        ssm.StringParameter(self, "BedrockBatchInferenceServiceRoleArn",
                            parameter_name=iac_config.BEDROCK_BATCH_INFERENCE_SERVICE_ROLE_PARAM_NAME,
                            string_value=service_role.role_arn,
                            description="ARN of the Bedrock service role for batch inference")
        
        submitter_role = _create_submitter_role(self, user_name=self.node.try_get_context("submitter_name") or \
                                                    iac_config.DEFAULT_BEDROCK_BATCH_INFERENCE_JOBS_SUBMITTER,
                                                bedrock_service_role=service_role.role_name)
        ssm.StringParameter(self, "BedrockBatchInferenceJobsSubmitterRoleArn",
                            parameter_name=iac_config.BEDROCK_SUBMITTER_ROLE_PARAM_NAME,
                            string_value=submitter_role.role_arn,
                            description="ARN of the role for submitting Bedrock's batch inference job")
        
        CfnOutput(self, "BedrockServiceRoleArn", value=service_role.role_arn)
        CfnOutput(self, "SubmitterRoleArn", value=submitter_role.role_arn)
