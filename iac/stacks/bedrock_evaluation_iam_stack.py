import iac_config
from aws_cdk import (
    Aws,
    Stack,
    aws_ssm as ssm,
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


class BedrockEvaluationIAMStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        s3_input_bucket = self.node.try_get_context("s3_input_bucket") \
            or iac_config.BEDROCK_EVALUATION_BUCKET_NAME
        s3_results_folder = self.node.try_get_context("s3_results_folder") \
            or iac_config.BEDROCK_EVALUATION_RESULTS_FOLDER
        
        policy = get_policy(self, 
                            "BedrockEvaluationPolicy", 
                            PermissionsType.BEDROCK_EVALUATION,
                            s3_input_bucket=s3_input_bucket,
                            s3_results_folder=s3_results_folder)
        principal = get_service_principal("bedrock.amazonaws.com", 
                                          arn_equals_conditions={
                                            "aws:SourceArn": [
                                                f"arn:aws:bedrock:us-east-1:{Aws.ACCOUNT_ID}:model-invocation-job/*",
                                                f"arn:aws:bedrock:us-east-1:{Aws.ACCOUNT_ID}:evaluation-job/*"
                                            ]
                                          })

        role = create_role(self,
                           iac_config.DEFAULT_BEDROCK_EVALUATION_SERVICE_ROLE, 
                           principal,
                           "Bedrock service role for creating and running model-as-judge evaluation jobs.",
                           policies=policy)
        ssm.StringParameter(self, "BedrockEvaluationServiceRoleArn",
                            parameter_name=iac_config.BEDROCK_EVALUATION_SERVICE_ROLE_PARAM_NAME,
                            string_value=role.role_arn,
                            description="ARN of the Bedrock service role for creating and running model-as-judge evaluation jobs.")
        
        CfnOutput(self, "EvaluationServiceRoleArn", value=role.role_arn)
        
        
