from aws_cdk import (
    Stack,
    aws_ec2 as ec2
)
from constructs import Construct
from custom_constructs.ec2_cuad_preprocessing import (
    CuadProcessingVpcConstruct,
    CuadPreprocessingEC2Construct
)


class CuadProcessingStack(Stack):

    def __init__(self, 
                 scope: Construct, 
                 construct_id: str, 
                 vpc: ec2.Vpc, 
                 **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        vpc = CuadProcessingVpcConstruct(self, "vpc_stack")
        ec2 = CuadPreprocessingEC2Construct(self, "ec2_cuad_preprocessing", vpc.vpc)
