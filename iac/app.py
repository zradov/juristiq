#!/usr/bin/env python3
import os
import aws_cdk as cdk
from stacks.app_stack import AppStack
from stacks.vpc_stack import VpcStack
from stacks.cuad_processing_stack import CuadProcessingStack


app = cdk.App()

vpc_stack = VpcStack(app, "VpcStack")
AppStack(app, "AppStack", vpc=vpc_stack.vpc)
CuadProcessingStack(app, "CuadProcessingStack", vpc=vpc_stack.vpc)

app.synth()
