#!/bin/bash
yum update -y
yum install -y python3.12.x86_64
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
/usr/bin/python3.12 get-pip.py
aws s3 cp s3://juristiq-code-24592-1273-31703-576 /home/ec2-user/code/ --recursive
aws s3 cp s3://juristiq-data/policies.json /home/ec2-user/code/
chown -R ec2-user:ec2-user /home/ec2-user/code
su - ec2-user -c 'python3.12 -m venv /home/ec2-user/code/.venv'
su - ec2-user -c 'source /home/ec2-user/code/.venv/bin/activate; \
pip install --upgrade pip; \
pip install -r /home/ec2-user/code/requirements.txt --no-cache-dir; \
pip install -r /home/ec2-user/code/dev-requirements.txt --no-cache-dir'
