import json
from typing import List, Dict, Generator
from botocore.client import BaseClient
from juristiq.cloud.utils import get_bedrock_client, get_s3_client
from juristiq.config.consts import TEXT_ENCODING
from juristiq.config.inference import S3_OUTPUT_FOLDER_FORMAT
    

def _get_s3_output_folder(job_info: Dict) -> str:
    """
    Returns the S3 path to the folder where the results, of an model evaluation job, are stored.

    Args:
        job_info: an object containing information about the evaluation job.

    Returns:
        S3 path the folder containing evaluation's job results. 
    """

    output_folder = job_info["outputDataConfig"]["s3Uri"]
    job_name = job_info["jobName"]
    job_uuid = job_info["jobArn"].split("/")[-1]
    model_id = job_info["inferenceConfig"]["models"][0]["bedrockModel"]["modelIdentifier"]
    task_type = job_info["evaluationConfig"]["automated"]["datasetMetricConfigs"][0]["taskType"]
    dataset = job_info["evaluationConfig"]["automated"]["datasetMetricConfigs"][0]["dataset"]["name"]

    return S3_OUTPUT_FOLDER_FORMAT.format(output_folder=output_folder,
                                          job_name=job_name,
                                          job_uuid=job_uuid,
                                          model_id=model_id,
                                          task_type=task_type,
                                          dataset=dataset)


def _get_evaluation_results_files(s3_client: BaseClient,
                                  s3_output_folder: str) -> List[str]:
    """
    Return information about S3 files containing the model evaluation results.

    Args:
        s3_client: boto3 S3 client instance.
        s3_output_folder: a path to the S3 folder where the results of the evaluation job are saved.

    Returns:
        a tuple with the bucket name and the S3 path to the folder with .jsonl files containing the 
        full response from the evaluation model including the evaluation scores.
    """
    s3_path_parts = s3_output_folder.removeprefix("s3://").split('/')
    bucket_name, prefix = s3_path_parts[0], "/".join(s3_path_parts[1:])
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    files = response.get("Contents", [])

    return bucket_name, files



def _get_evaluation_results(job_info: Dict, 
                            file_ext: str = ".jsonl") -> Generator[float, None, None]:
    """
    Returns evaluations scores from the files containing the evaluation model's full responses.

    Args:
        job_info: an object containing information about the evaluation job.
        file_ext: the extension of files that need to be selected.

    Returns:
        a Generator object containing the evaluation score. 
    """
    s3 = get_s3_client()
    s3_output_folder = _get_s3_output_folder(job_info)
    bucket_name, files = _get_evaluation_results_files(s3, s3_output_folder)

    for file in files:
        if file["Key"].endswith(file_ext):
            obj_res = s3.get_object(Bucket=bucket_name, Key=file["Key"])
            results = [json.loads(o) 
                       for o in obj_res.get("Body").read().decode(encoding=TEXT_ENCODING).split("\n")]
            # filter out results for which the evaluation model did not produce the evaluation score.
            results = [r for r in results 
                       if r["automatedEvaluationResult"]["scores"][0]["result"] is not None]

            for result in results:
                yield result["automatedEvaluationResult"]["scores"][0]["result"]


def get_evaluation_performance(job_name_filter: str) -> float:
    """
    Returns the overall score of the Bedrock's evaluation job.

    Args:
        job_name_filter: the string value that the job name needs to contain in order to be selected.

    Returns:
        an overall score for all selected evaluation jobs.
    """
    client = get_bedrock_client()
    res = client.list_evaluation_jobs(nameContains=job_name_filter)

    if not res["jobSummaries"]:
        raise Exception(f"No evaluation jobs found that contain the name '{job_name_filter}'.")
    
    completed_jobs = [j["jobArn"] for j in res["jobSummaries"] if j["status"] == "Completed"]

    if not completed_jobs:
        raise Exception(f"No completed evaluation jobs found that contain the name '{job_name_filter}'.")
    
    results = []

    for job_arn in completed_jobs:
        job_info = client.get_evaluation_job(jobIdentifier=job_arn)
        results.extend(_get_evaluation_results(job_info))

    overall_score = sum(results) / len(results)

    return overall_score
