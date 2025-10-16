import os
import sys
import time
import uuid
import boto3
import argparse
from iac import iac_config
import juristiq.config.cloud as cloud_config
from juristiq.inference.models import ModelName


def parse_args():

    parser = argparse.ArgumentParser(description="Bedrock Batch Inference Job Creation")

    parser.add_argument("-i", "--input-bucket",
                        required=False,
                        type=str,
                        default=iac_config.BEDROCK_BATCH_INFERENCE_INPUT_BUCKET_NAME,
                        help="A path to the AWS S3 bucket containing .jsonl files.")
    parser.add_argument("-o", "--output-bucket",
                        required=False,
                        type=str,
                        default=iac_config.BEDROCK_BATCH_INFERENCE_OUTPUT_BUCKET_NAME,
                        help="A local to the AWS S3 bucket where the batch inference results will be stored.")
    parser.add_argument("-m", "--model-name",
                        required=False,
                        default=ModelName.NOVA_LITE.value,
                        help="A name of the Bedrock LLM that will process prompts.")

    args = parser.parse_args()

    return args


def _get_batch_inference_config(key: str,
                                s3_bucket: str) -> dict:
    
    data_config = {
        key: {
            "s3Uri": f"s3://{s3_bucket}/"
        }
    }
    
    return data_config


def _get_bedrock_client():
    """
    Creates and returns a Bedrock client.

    Returns:
        A boto3 Bedrock client.
    """
    session = boto3.Session(profile_name=cloud_config.PROFILE_NAME)
    sts_client = session.client("sts")
    ssm_client = session.client("ssm")

    param = ssm_client.get_parameter(Name=iac_config.BEDROCK_SUBMITTER_ROLE_PARAM_NAME)
    submitter_role_arn = param["Parameter"]["Value"]
    
    res = sts_client.assume_role(RoleArn=submitter_role_arn,
                                 RoleSessionName="BedrockJobSession")
    
    creds = res["Credentials"]
    access_key = creds["AccessKeyId"]
    secret_key = creds["SecretAccessKey"]
    session_token = creds["SessionToken"]

    bedrock_client = boto3.client(
        "bedrock",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        aws_session_token=session_token,
        region_name=session.region_name
    )

    return bedrock_client


def main(s3_input_bucket: str,
         s3_output_bucket: str,
         model_name: str,
         refresh_interval: int=2) -> None:
    """
    Creates a Bedrock batch inference job and waits for it to finish.

    Args:
        s3_input_bucket: An S3 bucket containing input .jsonl files.
        s3_output_bucket: An S3 bucket where the output files will be stored.
        model_name: A name of the Bedrock LLM that will process prompts.
        refresh_interval: An interval in seconds between status checks.    
    """
    session = boto3.Session(profile_name=cloud_config.PROFILE_NAME)
    ssm_client = session.client("ssm")
    bedrock_client = _get_bedrock_client()
    
    input_data_config = _get_batch_inference_config("s3InputDataConfig", s3_input_bucket)
    output_data_config = _get_batch_inference_config("s3OutputDataConfig", s3_output_bucket)
    param = ssm_client.get_parameter(Name=iac_config.BEDROCK_SERVICE_ROLE_PARAM_NAME)
    bedrock_service_role_name = param["Parameter"]["Value"]

    response = bedrock_client.create_model_invocation_job(
        roleArn=bedrock_service_role_name,
        modelId=model_name,
        jobName=f"{model_name}-eval-{str(uuid.uuid4())[:8]}-new",
        inputDataConfig=input_data_config,
        outputDataConfig=output_data_config
    )

    job_arn = response.get("jobArn")

    print(f"Waiting for the job {job_arn} to finish ...")

    while True:

        job_info = bedrock_client.get_model_invocation_job(jobIdentifier=job_arn)
        status = job_info["status"]
        elapsed_time = job_info["endTime"] - job_info["submitTime"]
        lines = [
            f"Status: {status}",
            f"Elapsed time: {elapsed_time}"
        ]

        print("\n".join(lines))
        time.sleep(refresh_interval)
        sys.stdout.write(f"\033[{len(lines)}A")


if __name__ == "__main__":
    
    args = parse_args()

    main(args.input_bucket, args.output_bucket, args.model_name)
