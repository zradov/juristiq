#!/usr/bin/env python3
import aws_cdk as cdk
from stacks.app_stack import AppStack
from stacks.vpc_stack import VpcStack
from stacks.storage_stack import StorageStack
from stacks.cuad_processing_stack import CuadProcessingStack
from stacks.bedrock_batch_iam_stack import BedrockBatchIAMStack
from stacks.bedrock_evaluation_iam_stack import BedrockEvaluationIAMStack


app = cdk.App()

vpc_stack = VpcStack(app, "VpcStack")
AppStack(app, "AppStack", vpc=vpc_stack.vpc)
CuadProcessingStack(app, "CuadProcessingStack", vpc=vpc_stack.vpc)
BedrockBatchIAMStack(app, "BedrockBatchIAMStack")
BedrockEvaluationIAMStack(app, "BedrockEvaluationIAMStack")
StorageStack(app, "StorageStack")

app.synth()
