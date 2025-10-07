# Overview

The folder contains various IaC code for creating and configuring stacks and individual resources. 

## Required policies

$policyContent = (Get-Content ".\iac\policies\cdk_bootstrap_permissions_template.json").Replace("{Aws.ACCOUNT_ID}", $accountId).Replace("{Aws.REGION}", $region)
$res = aws iam create-policy --policy-name CdkBootstrap --policy-document "$policyContent"
$policyData = ConvertFrom-Json $($res -join "")
aws iam attach-user-policy --user-name USER_NAME --policy-arn $policyData.Policy.Arn
aws iam create-policy --policy-name AllowVpcManagement --policy-document file://

aws iam create-policy --policy-name AllowCDKDevelopment --policy-document file://cdk_permissions_policy.json
aws iam-

## Workflow

cdk  bootstrap --profile USER_PROFILE_NAME
cdk deploy --require-approval never --profile USER_PROFILE_NAME -O output.json

# List available stacks for deployment
cdk list 

## EC2 Cuad Preprocessing Stack

If needed create key pair by using the command:
```
aws ec2 create-key-pair \
    --key-name ec2_cuad_processing \
    --key-type rsa \
    --key-format pem \
    --query "CuadProcessing" \
    --output text > ec2_cuad_processing.pem
```
and then reference it through command line as a context variable when creating the CUAD processing stack:
```
cdk  deploy CuadProcessingStack -c key_name=KEY_PAIR_NAME --profile dev
```

cdk  deploy CuadProcessingStack --profile dev

> The SSH access to the created EC2 instance can be further restricted by passing the **ssh_allowed_ip** context variable
  when creating the CUAD processing EC2 instance:

  ```
  cdk  deploy CuadProcessingStack -c ssh_allowed_ip=SOME_IP_ADDRESS/SUBNET_MASK --profile dev

