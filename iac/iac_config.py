import os
from typing import Final
from pathlib import Path


_IAC_FOLDER_PATH = Path(os.path.abspath(__name__)).parent
_DATA_FOLDER =  _IAC_FOLDER_PATH.parent / "data"
_POLICIES_PATH = _IAC_FOLDER_PATH / "policies"
S3_BUCKET_NAME_PREFIX = os.getenv("S3_BUCKET_NAME_PREFIX", "juristiq")

# All data related to the Juristiq project.
JURISTIQ_DATA_BUCKET_NAME = f"{S3_BUCKET_NAME_PREFIX}-data"
# Preprocessed data ready for ML model training.
ML_DATA_BUCKET_NAME: Final[str] = f"{S3_BUCKET_NAME_PREFIX}-ml-data"
# Separate annotations files for each contract.
CUAD_CHUNKED_ANNOTS_BUCKET_NAME: Final[str] = f"{S3_BUCKET_NAME_PREFIX}-cuad-chunked-annots"
# Files containing contracts text.
CUAD_FULL_CONTRACTS_BUCKET_NAME: Final[str] = f"{S3_BUCKET_NAME_PREFIX}-cuad-full-contracts"
# Transformed CUAD annotations augmented with additional field required for LLM fine-tunning.
CUAD_TRANSFORMED_BUCKET_NAME: Final[str] = f"{S3_BUCKET_NAME_PREFIX}-cuad-transformed"
# Augmented CUAD annotations updated with policy compliance review from an LLM model playing the legal AI assistant role.
CUAD_REVIEWED_BUCKET_NAME: Final[str] = f"{S3_BUCKET_NAME_PREFIX}-cuad-reviewed"
# The project code and other automation scripts
JURISTIQ_CODE_BUCKET_NAME = "juristiq-code-24592-1273-31703-576"
# The name of the S3 bucket where the input files, for the Bedrock batch inference, are stored.
BEDROCK_BATCH_INFERENCE_INPUT_BUCKET_NAME = f"{S3_BUCKET_NAME_PREFIX}-bedrock-batch-inference-input"
# The name of the S3 bucket where the output of the Bedrock batch inference is stored.
BEDROCK_BATCH_INFERENCE_OUTPUT_BUCKET_NAME = f"{S3_BUCKET_NAME_PREFIX}-bedrock-batch-inference-output"
# The name of the S3 bucket where the input files and the results of the Bedrock model evaluation are stored.
BEDROCK_EVALUATION_BUCKET_NAME = f"{S3_BUCKET_NAME_PREFIX}-bedrock-evaluation"
# The path to the S3 bucket's folder where the results of the Bedrock evaluation are stored.
BEDROCK_EVALUATION_RESULTS_FOLDER = f"{BEDROCK_EVALUATION_BUCKET_NAME}/results/"

DEFAULT_BEDROCK_EVALUATION_SERVICE_ROLE="Amazon-Bedrock-IAM-Role-Judge-LLM-Evaluation"
# The original CUAD annotations path.
CUAD_ANNOTS_PATH: Final[str] = os.path.join(_DATA_FOLDER, "CUAD_v1.json")
# IAM policy for VPC management. 
VPC_MANAGEMENT_POLICY_PATH = _POLICIES_PATH / "vpc_management.json"
# IAM policy attached to the EC2 instance used to process and augment the CUAD .json annotations.
CUAD_PROCESSING_POLICY_PATH = _POLICIES_PATH / "ec2_to_s3_cuad_processing_permissions.json"
# Policy containing Bedrock service role permissions for running batch inference.
BEDROCK_BATCH_INFERENCE_SERVICE_ROLE_TEMPLATE_PATH = _POLICIES_PATH / "bedrock_batch_inference_service_role_template.json"
# Bedrock model invocation permissions
BEDROCK_BATCH_INFERENCE_MODEL_INVOCATION_TEMPLATE_PATH = _POLICIES_PATH / "bedrock_batch_inference_model_invocation_template.json"
# Managed Bedrock policy for submitting jobs and passing service role
BEDROCK_BATCH_INFERENCE_SUBMITTER_TEMPLATE_PATH = _POLICIES_PATH / "bedrock_batch_inference_submitter_template.json"
# Managed Bedrock policy for creating and running Bedrock model evaluation jobs.
BEDROCK_MODEL_AS_JUDGE_EVALUATION_TEMPLATE_PATH = _POLICIES_PATH / "bedrock_model_as_judge_evaluation_template.json"
# User data script run at the boot time of the EC2 instance used to process and augment the CUAD .json annotations.
EC2_CUAD_PROCESSING_USER_DATA_PATH = _IAC_FOLDER_PATH / "user_data" / "ec2_instance_bootstrap.sh"
DEFAULT_BEDROCK_BATCH_INFERENCE_MODEL = "amazon.nova-lite-v1:0"
# The user name of the user who will creates a batch inference jobs to invoke a LLM models.
DEFAULT_BEDROCK_BATCH_INFERENCE_JOBS_SUBMITTER = "juristiqdev"
# The name of the AWS System Manager parameter containing the name of the Bedrock's service role 
# for running the batch inference jobs.
BEDROCK_BATCH_INFERENCE_SERVICE_ROLE_PARAM_NAME = "/bedrock/batch-inference-service-role-arn"
# The name of the AWS System Manager parameter containing the name of the user role 
# who will be creating and running Bedrock's batch inference jobs.
BEDROCK_SUBMITTER_ROLE_PARAM_NAME = "/bedrock/batch-inference-jobs-submitter-role-arn"
# The name of the AWS System Manager parameter containing the name of the Bedrock's service role 
# for running the evaluation jobs.
BEDROCK_EVALUATION_SERVICE_ROLE_PARAM_NAME = "/bedrock/evaluation-service-role-arn"


