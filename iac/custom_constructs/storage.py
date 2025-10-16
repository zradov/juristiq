from aws_cdk import (
    RemovalPolicy,
    aws_s3 as s3,
    Tags
)
from constructs import Construct


class StorageConstruct(Construct):
    """
    A construct for creating an S3 bucket with optional tags.
    """

    def __init__(self, 
                 scope: Construct, 
                 id: str, 
                 bucket_name: str, 
                 tags: dict=None, 
                 **kwargs):
        """
        Initialize the StorageConstruct.

        Args:
            scope (Construct): The scope in which this construct is defined.
            id (str): The scoped construct ID.
            bucket_name (str): The name of the S3 bucket to create or reference.
            tags (dict, optional): A dictionary of tags to apply to the bucket. Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(scope, id, **kwargs)

        try:
            self.bucket = s3.Bucket.from_bucket_name(self, "Bucket", bucket_name)
        except:
            self.bucket = s3.Bucket(self, 
                            "Bucket", 
                            bucket_name, 
                            auto_delete_objects=True,
                            removal_policy=RemovalPolicy.DESTROY, 
                            public_read_access=False)
        
        if not tags:
            tags = {"Purpose": "Test"}
        
        if tags:
            for key, value in tags.items():
                Tags.of(self.bucket).add(key, value)
