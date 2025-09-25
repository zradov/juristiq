import json
import iac_config
from aws_cdk import CfnParameter, Tags
from aws_cdk.aws_ec2 import (
    Port, 
    Peer, 
    Vpc,
    SubnetType,
    AmazonLinuxGeneration,
    AmazonLinuxCpuType,
    Instance,
    AmazonLinuxImage,
    SecurityGroup,
    InstanceClass,
    InstanceSize,
    InstanceType,
    IpAddresses,
    SubnetSelection,
    SubnetConfiguration,
    SubnetType
)
from aws_cdk.aws_iam import (
    Role, 
    ServicePrincipal, 
    Policy, 
    PolicyDocument
)
from constructs import Construct


def _load_user_data(path: str) -> str:

    with open(path, encoding="utf-8") as fp:
        return fp.read()


def _load_policy(path: str) -> dict:

    with open(path) as fp:
        return json.load(fp)


class CuadProcessingVpcConstruct(Construct):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.output_vpc = Vpc(self, 
                           "VpcCuadProcessing",
                           ip_addresses=IpAddresses.cidr("10.0.0.0/16"),
                           max_azs=2,
                           subnet_configuration=[
                               SubnetConfiguration(name="public", subnet_type=SubnetType.PUBLIC, cidr_mask=24)
                           ])
        Tags.of(self.output_vpc).add("Purpose", "Test")
    
    
    @property
    def vpc(self) -> Vpc:
        return self.output_vpc
    

class CuadPreprocessingEC2Construct(Construct):

    def __init__(self, scope: Construct, id: str, vpc: Vpc, **kwargs):
        super().__init__(scope, id, **kwargs)

        ec2_role_conditions = {
            "StringEquals": {
                "aws:RequestTag/Purpose": "test"
            },
            "ForAllValues:StringEquals": {
                 "aws:ResourceTag/Purpose": "Test"
            }
        }
        ec2_role = Role(self, "EC2ToS3Role", 
                        assumed_by=ServicePrincipal("ec2.amazonaws.com", 
                                                    conditions=ec2_role_conditions))
        policy_doc = PolicyDocument.from_json(_load_policy(iac_config.CUAD_PROCESSING_POLICY_PATH))
        ec2_role.attach_inline_policy(
            Policy(self, "vpc_management_policy", document=policy_doc))
        ec2_sec_group = SecurityGroup(self, 
                                      "ssh_sg", 
                                      vpc=vpc,
                                      description="Allow Inbound SSH traffic.",
                                      allow_all_outbound=True)
        ssh_allowed_ip = self.node.try_get_context("ssh_allowed_ip") or "0.0.0.0/0"
        ec2_sec_group.add_ingress_rule(
            Peer.ipv4(ssh_allowed_ip),
            Port.tcp(22)
        )
        ami = AmazonLinuxImage(
            generation=AmazonLinuxGeneration.AMAZON_LINUX_2023,
            cpu_type=AmazonLinuxCpuType.X86_64
        )
        key_pair_name = self.node.try_get_context("key_name") or None
        ec2_instance = Instance(self, 
                                "ec2_cuad_preprocessing",
                                vpc=vpc,
                                vpc_subnets=SubnetSelection(
                                    subnet_type=SubnetType.PUBLIC
                                ),
                                instance_type=InstanceType.of(InstanceClass.T3, InstanceSize.XLARGE),
                                machine_image=ami,
                                role=ec2_role,
                                key_name=key_pair_name,
                                security_group=ec2_sec_group)
        ec2_user_data = _load_user_data(iac_config.EC2_CUAD_PROCESSING_USER_DATA_PATH)
        ec2_instance.add_user_data(ec2_user_data)
        Tags.of(ec2_instance).add("Purpose", "Test")
