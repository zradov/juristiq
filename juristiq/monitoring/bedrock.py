import os
import sys
import time
import boto3
from datetime import (
    datetime, 
    timezone, 
    timedelta
)
from botocore.client import BaseClient
from juristiq.inference.models import ModelName
from typing import NamedTuple, List, Dict, Tuple
from juristiq.monitoring.costs_calculator import calculate_costs


class TokensCost(NamedTuple):
    """Cost per input and output tokens."""
    input_tokens: float
    output_tokens: float


class TokensMetrics(NamedTuple):
    """Metrics for total input and output tokens."""
    total_input_tokens: int
    total_output_tokens: int


"""Costs per 1000 tokens for different models."""
_COSTS_PER_1000_TOKENS = {
    ModelName.NOVA_LITE: TokensCost(input_tokens=0.00006, 
                                    output_tokens=0.00024)
}


def _get_client(aws_service: str="cloudwatch") -> BaseClient:
    """
    Initializes and returns a boto3 client for the specified AWS service.

    Args:
        aws_service: The name of the AWS service to create a client for.

    Returns:
        A boto3 client for the specified AWS service.
    """
    
    session = boto3.Session(profile_name=os.environ["AWS_PROFILE_NAME"])
    client = session.client(aws_service)
    
    return client


def _get_metrics_data_queries(model_id: str,
                              namespace: str="AWS/Bedrock",
                              metrics: Tuple=("InputTokenCount", "OutputTokenCount"),
                              period: int=60) -> List[Dict]:
    """
    Returns a list metric queries.

    Args:
        model_id: an ID of LLM model.
        namespace: the namespace of the metric.
        metrics: the metrics names.
        period: the granularity, in seconds, of the returned data points.

    Returns:
        a list of metric queries.
    """
    queries = []

    for i, m in enumerate(metrics):
        queries.append({
            "Id": f"m{i}",
            "MetricStat": {
                "Metric": {
                    "Namespace": namespace,
                    "MetricName": m,
                    "Dimensions": [
                        {"Name": "ModelId", "Value": model_id}
                    ],
                },
                "Period": period,
                "Stat": "Sum"
            },
            "ReturnData": True
        })
    
    return queries


def _process_metric_data_results(metric_data_results: List[Dict],
                                 metrics: Tuple=("InputTokenCount", "OutputTokenCount")) -> Dict[str, Dict]:
    """
    Processes the metric data results from AWS CloudWatch and organizes them into a dictionary.

    Args:
        metric_data_results: A list of metric data results from AWS CloudWatch.

    Returns:
        A dictionary where keys are metric names and values are dictionaries containing 
        timestamps, values for each time period and the sum of all values.
    """
    results = {
        "InputTokenCount": {},
        "OutputTokenCount": {}
    }

    for mquery in metric_data_results:
        mid = mquery["Id"]
        metric_name = metrics[int(mid[1:])]
        if metric_name in results:
            results[metric_name] = {
                "Timestamps": mquery.get("Timestamps", []),
                "Values": mquery.get("Values", []),
                "Total": sum(mquery.get("Values", []))
            }

    return results


def fetch_token_metrics(model_id: str, 
                        start_time: datetime, 
                        end_time: datetime,
                        period: int) -> TokensMetrics:
    """
    Fetches token metrics (input and output token counts) from AWS CloudWatch 
    for a specified model over a given time range.

    Args:
        model_id: The ID of the model to fetch metrics for.
        start_time: The start time for the metrics retrieval (UTC).
        end_time: The end time for the metrics retrieval (UTC).
        period: The granularity, in seconds, of the returned data points.

    Returns:
        A TokensMetrics named tuple containing total input and output tokens.
    """
    client = _get_client()
    
    queries = _get_metrics_data_queries(model_id, period=period)
    
    resp = client.get_metric_data(
        MetricDataQueries=queries,
        StartTime=start_time,
        EndTime=end_time,
        ScanBy="TimestampAscending"
    )
    
    results = _process_metric_data_results(resp.get("MetricDataResults", []))

    return TokensMetrics(total_input_tokens=results["InputTokenCount"]["Total"], 
                         total_output_tokens=results["OutputTokenCount"]["Total"])


def show_token_metrics(model: ModelName,
                       start: datetime=None, 
                       end: datetime=None,
                       loop: bool=False,
                       delta_hours: int=10,
                       refresh_interval: int=2) -> None:
    """
    Displays token metrics and associated costs for a specified model over a given time range.

    Args:
        model: The model name to fetch metrics for.
        start: The start time for the metrics retrieval (UTC). If None, defaults to current time minus delta_hours.
        end: The end time for the metrics retrieval (UTC). If None, defaults to current time.
        loop: If True, continuously refreshes and displays the metrics at the specified interval.
        delta_hours: The number of hours to look back if start time is not provided.
        refresh_interval: The interval in seconds at which to refresh the metrics if loop is True.
    """
    print()
    
    while True:

        end = end if end else datetime.now(timezone.utc)
        start = start if start else end - timedelta(hours=delta_hours)
        tokens_metrics = fetch_token_metrics(model.value, start, end, period=60)
        current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        tokens_cost = calculate_costs(model, tokens_metrics)

        lines = [
            f"Current time: {current_time}",
            "",
            "Tokens metrics:",
            f"=> Total input tokens: {tokens_metrics.total_input_tokens}",
            f"=> Total output tokens: {tokens_metrics.total_output_tokens}",
            "",
            "Costs:",
            f"=> Input tokens: {tokens_cost.input:.4f}",
            f"=> Output tokens: {tokens_cost.output:.4f}",
            f"=> Total: {tokens_cost.input + tokens_cost.output:.4f}"
        ]
             
        print("\n".join(lines))
        time.sleep(refresh_interval)
        sys.stdout.write(f"\033[{len(lines)}A")

        if not loop:
            break

    print("\n" * (len(lines) - 1))


if __name__ == "__main__":

    show_token_metrics(ModelName.NOVA_LITE, loop=True, delta_hours=48)
        