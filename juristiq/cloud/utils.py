import os
import boto3
import aioboto3
from botocore.client import BaseClient


def get_client(service_name: str, enable_async: bool=False) -> BaseClient:
    """
    Get a boto3 client for the specified AWS service using the profile
    name from the environment variable 'AWS_PROFILE_NAME'.

    Args:
        service_name: The name of the AWS service.
        enable_async: Should client's requests run asynchronously.

    Returns:
        A boto3 client for the specified service.
    """
    session_cls = aioboto3.Session if enable_async else boto3.Session
    session = session_cls(profile_name=os.environ["AWS_PROFILE_NAME"])
    client = session.client(service_name, region_name=session.region_name)

    return client


def get_bedrock_runtime_client(enable_async: bool=False) ->  BaseClient:
    
    return get_client("bedrock-runtime", enable_async)


def get_bedrock_client(enable_async: bool=False) -> BaseClient:

    return get_client("bedrock", enable_async)


def get_s3_client() -> BaseClient:

    return get_client("s3")


def get_account_id() -> str:
    """
    Get the AWS account ID associated with the current credentials.

    Returns:
        The AWS account ID as a string.
    """
    client = get_client("sts")
    identity = client.get_caller_identity()

    return identity["Account"]

        