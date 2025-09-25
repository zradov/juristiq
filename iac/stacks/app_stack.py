from aws_cdk import (
    Stack,
    aws_ec2 as ec2
)
from constructs import Construct
from custom_constructs.buckets import DataS3Bucket


class AppStack(Stack):

    def __init__(self, 
                 scope: Construct, 
                 construct_id: str, 
                 vpc: ec2.Vpc, 
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        s3_bucket = DataS3Bucket(self, "DataS3Bucket")
