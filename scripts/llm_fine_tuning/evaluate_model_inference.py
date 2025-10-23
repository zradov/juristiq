import time
import uuid
import logging
from botocore.client import BaseClient
from juristiq.inference.models import ModelName
from argparse import ArgumentParser, ArgumentTypeError
from juristiq.config.logging_config import configure_logging
from juristiq.data_preprocessing.annots import load_data_from_jsonl
from juristiq.cloud.utils import get_bedrock_client, get_account_id
from juristiq.config.inference import (
    DEFAULT_JUDGE_INFERENCE_PARAMS, 
    CUSTOM_EVALUATION_METRIC_NAME,
    CUSTOM_DATASET_NAME
)
from juristiq.config.templates import BEDROCK_MODEL_EVALUATION_CUSTOM_METRICS_PROMPT


configure_logging()
logger = logging.getLogger(__name__)


def s3_file_path(value: str) -> str:

    if not value.startswith("s3://"):
        return f"s3://{value}"
    
    return value


def s3_bucket_folder(value: str) -> str:

    if not value.startswith("s3://"):
        value = f"s3://{value}"

    if not value.endswith("/"):
        value += "/"

    return value


def model_type(value: str) -> str:

    if value not in ModelName:
        raise ArgumentTypeError(f"The model id '{value}' is not supported.")

    return value


def _parse_args():

    parser = ArgumentParser(description="Bedrock Inference Metrics Evaluation")

    parser.add_argument("-j", "--job-name",
                        help="The name of the evaluation job.")
    parser.add_argument("-i", "--input-file",
                        required=False,
                        type=s3_file_path,
                        help="S3 path to the .jsonl file containing prompt samples.")
    parser.add_argument("-o", "--output-folder",
                        type=s3_bucket_folder,
                        help="The path to the AWS S3 folder where the evaluation results will be stored.")
    parser.add_argument("-g", "--inference-model-id",
                        required=False,
                        default=ModelName.NOVA_LITE.value,
                        type=model_type,
                        help="An identifier of the Bedrock LLM that will be used for inference.")
    parser.add_argument("-e", "--evaluation-model-id",
                        default=ModelName.NOVA_PRO.value,
                        type=model_type,
                        help="An identifier of the Bedrock LLM that will be used for evaluation.")
    parser.add_argument("-r", "--service-role",
                        help="A Bedrock service role required for creating and running evaluation jobs.")


    args = parser.parse_args()

    return args


def _get_evaluation_config(input_file_path: str,
                           evaluation_model_id: str) -> dict:

    config = {
        "automated": {
            "datasetMetricConfigs": [
                {
                    "taskType": "General",
                    "dataset": {
                        "name": CUSTOM_DATASET_NAME,
                        "datasetLocation": {
                            "s3Uri": input_file_path
                        }
                    },
                    "metricNames": [CUSTOM_EVALUATION_METRIC_NAME]
                }
            ],
            "customMetricConfig": {
                "customMetrics": [
                    {
                        "customMetricDefinition": {
                            "name": CUSTOM_EVALUATION_METRIC_NAME,
                            "instructions": BEDROCK_MODEL_EVALUATION_CUSTOM_METRICS_PROMPT.read_text(encoding="utf8"),
                            "ratingScale": [
                                {
                                    "definition": "Excellent",
                                    "value": {
                                        "floatValue": 3
                                    }
                                },
                                    {
                                    "definition": "Good",
                                    "value": {
                                        "floatValue": 2
                                    }
                                },
                                    {
                                    "definition": "Poor",
                                    "value": {
                                        "floatValue": 1
                                    }
                                }
                            ]
                        }
                    },
                ],
                "evaluatorModelConfig": {
                    "bedrockEvaluatorModels": [
                        {
                            "modelIdentifier": evaluation_model_id
                        }
                    ]
                }
            },
            "evaluatorModelConfig": {
                "bedrockEvaluatorModels": [
                    {
                        "modelIdentifier": evaluation_model_id
                    }
                ]
            }
        },  
    }

    return config


def _get_inference_config(inference_model_id: str) -> str:

    config = {
        "models": [
            {
                "bedrockModel": {
                    "modelIdentifier": inference_model_id,
                    "inferenceParams": f'{{"inferenceConfig": {DEFAULT_JUDGE_INFERENCE_PARAMS}}}'
                }
            },
        ]
    }

    return config


def _get_unique_job_name(bedrock: BaseClient,
                         job_name: str,
                         max_job_name_len: int=60) -> str:
    
    response = bedrock.list_evaluation_jobs(nameContains=job_name)

    if response["jobSummaries"]:
        existing_jobs = list(filter(lambda i: i["jobName"] == job_name, response["jobSummaries"]))

        if existing_jobs:
            name = f"{job_name}-{uuid.uuid4().hex}"
            return name[:max_job_name_len]

    return job_name


def _create_evaluation_job(bedrock: BaseClient,
                           job_name: str,
                           input_file_path: str,
                           output_folder_path: str,
                           inference_model_id: str,
                           evaluation_model_id: str,
                           service_role_name: str) -> dict:

    unique_job_name = _get_unique_job_name(bedrock, job_name)

    logger.info(f"New unique evaluation job name: {unique_job_name}")

    response = bedrock.create_evaluation_job(
        jobName=unique_job_name,
        roleArn=f"arn:aws:iam::{get_account_id()}:role/{service_role_name}",
        evaluationConfig=_get_evaluation_config(input_file_path, evaluation_model_id),
        inferenceConfig=_get_inference_config(inference_model_id),
        outputDataConfig={ "s3Uri": output_folder_path },
        jobDescription="Run inference with Nova Lite and evaluate outputs with Nova Pro."
    )

    return response


def check_job_status(client, job_arn) -> str:

    try:
        response = client.get_evaluation_job(jobIdentifier=job_arn)
        return response["status"]
    except Exception as ex:
        print(ex)
        return "ERROR"
    

def main(job_name: str,
         input_file_path: str,
         output_folder_path: str,
         inference_model_id: str,
         evaluation_model_id: str,
         service_role_name: str):
    
    client = get_bedrock_client()
    
    response = _create_evaluation_job(client,
                                      job_name,
                                      input_file_path, 
                                      output_folder_path, 
                                      inference_model_id, 
                                      evaluation_model_id,
                                      service_role_name)
    
    job_status = "InProgress"
    logger.info(f"Job '{response['jobArn']}' has started.")
    while job_status == "InProgress":
        job_status = check_job_status(client, response["jobArn"])
        time.sleep(5)

    logger.info(f"Evaluation job '{response['jobArn']}' is completed.")


if __name__ == "__main__":

    args = _parse_args()

    main(args.job_name,
         args.input_file,
         args.output_folder,
         args.inference_model_id, 
         args.evaluation_model_id,
         args.service_role)
