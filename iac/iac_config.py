import os
from typing import Final
from pathlib import Path

_SCRIPT_PATH = Path(os.path.abspath(__name__)).parent
_DATA_FOLDER =  os.path.join(_SCRIPT_PATH, "..", "data")
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
CUAD_FINAL_BUCKET_NAME: Final[str] = f"{S3_BUCKET_NAME_PREFIX}-cuad-final"
# The project code and other automation scripts
JURISTIQ_CODE_BUCKET_NAME = "juristiq-code-24592-1273-31703-576"
CUAD_ANNOTS_PATH: Final[str] = os.path.join(_DATA_FOLDER, "CUAD_v1.json")
# IAM policy for VPC management. 
VPC_MANAGEMENT_POLICY_PATH = os.path.join(_SCRIPT_PATH, "policies", "vpc_management.json")
# IAM policy attached to the EC2 instance used to process and augment the CUAD .json annotations.
CUAD_PROCESSING_POLICY_PATH = os.path.join(_SCRIPT_PATH, "policies", "ec2_to_s3_cuad_processing_permissions.json")
# User data script run at the boot time of the EC2 instance used to process and augment the CUAD .json annotations.
EC2_CUAD_PROCESSING_USER_DATA_PATH = os.path.join(_SCRIPT_PATH, "user_data", "ec2_instance_bootstrap.sh")


