import os
import iac_config
from aws_cdk import (
    RemovalPolicy,
    aws_s3 as s3,
    aws_s3_deployment as s3_deploy,
    Tags
)
from pathlib import Path
from zipfile import ZipFile
from constructs import Construct


def _create_bucket(scope: Construct, id: str, name: str) -> s3.Bucket:
    
    bucket = s3.Bucket(scope, id, bucket_name=name, auto_delete_objects=True,
                       removal_policy=RemovalPolicy.DESTROY, public_read_access=False)
    Tags.of(bucket).add("Purpose", "Test")
    return bucket


class DataS3BucketConstruct(Construct):

    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        juristiq_data_bucket = _create_bucket(self, "s3_data_bucket", iac_config.JURISTIQ_DATA_BUCKET_NAME)
        cuad_transformed_bucket =_create_bucket(self, "s3_cuad_transformed_bucket", iac_config.CUAD_TRANSFORMED_BUCKET_NAME)
        cuad_reviewed_bucket = _create_bucket(self, "s3_cuad_final_bucket", iac_config.CUAD_REVIEWED_BUCKET_NAME)
        ml_data_bucket = _create_bucket(self, "s3_ml_data", iac_config.ML_DATA_BUCKET_NAME)
        code_bucket = _create_bucket(self, "code", iac_config.JURISTIQ_CODE_BUCKET_NAME)

        s3_deploy.BucketDeployment(self, 
                                   "s3_ml_data_deploy", 
                                   sources=[
                                        s3_deploy.Source.data("cuad-finetuning/train/empty.txt", data=" "),
                                        s3_deploy.Source.data("cuad-finetuning/validation/empty.txt", data=" "),
                                        s3_deploy.Source.data("cuad-finetuning/test/empty.txt", data=" ")
                                    ],
                                    destination_bucket=ml_data_bucket)
        s3_deploy.BucketDeployment(self, 
                                   "s3_cuad_annots", 
                                   sources=[s3_deploy.Source.data(os.path.basename(iac_config.CUAD_ANNOTS_PATH), 
                                                                  iac_config.CUAD_ANNOTS_PATH)],
                                   destination_bucket=juristiq_data_bucket)
        