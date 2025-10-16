import os
import iac_config
from aws_cdk import (
    Stack,
    aws_s3_deployment as s3_deploy,
)
from constructs import Construct
from custom_constructs.storage import StorageConstruct


class StorageStack(Stack):

    def __init__(self, 
                 scope: Construct, 
                 construct_id: str,
                 **kwargs):
        super().__init__(scope, construct_id, **kwargs)

        juristiq_data_bucket = StorageConstruct(self, 
                                                "S3DataBucket", 
                                                iac_config.JURISTIQ_DATA_BUCKET_NAME)
        cuad_transformed_bucket = StorageConstruct(self, 
                                                   "S3CuadTransformedBucket", 
                                                   iac_config.CUAD_TRANSFORMED_BUCKET_NAME)
        cuad_reviewed_bucket = StorageConstruct(self, 
                                                "S3CuadFinalBucket", 
                                                iac_config.CUAD_REVIEWED_BUCKET_NAME)
        ml_data_bucket = StorageConstruct(self, "S3MLData", iac_config.ML_DATA_BUCKET_NAME)
        code_bucket = StorageConstruct(self, "Code", iac_config.JURISTIQ_CODE_BUCKET_NAME)
        batch_inference_input = StorageConstruct(self, 
                                                 "S3BedrockBatchInferenceInput", 
                                                 iac_config.BEDROCK_BATCH_INFERENCE_INPUT_BUCKET_NAME)
        batch_inference_output = StorageConstruct(self, 
                                                  "S3BedrockBatchInferenceOutput", 
                                                  iac_config.BEDROCK_BATCH_INFERENCE_OUTPUT_BUCKET_NAME)

        s3_deploy.BucketDeployment(self, 
                                   "S3MLDataDeploy", 
                                   sources=[
                                        s3_deploy.Source.data("cuad-finetuning/train/empty.txt", data=" "),
                                        s3_deploy.Source.data("cuad-finetuning/validation/empty.txt", data=" "),
                                        s3_deploy.Source.data("cuad-finetuning/test/empty.txt", data=" ")
                                    ],
                                    destination_bucket=ml_data_bucket.bucket)
        s3_deploy.BucketDeployment(self, 
                                   "S3CuadAnnots", 
                                   sources=[s3_deploy.Source.data(os.path.basename(iac_config.CUAD_ANNOTS_PATH), 
                                                                  iac_config.CUAD_ANNOTS_PATH)],
                                   destination_bucket=juristiq_data_bucket.bucket)
        
