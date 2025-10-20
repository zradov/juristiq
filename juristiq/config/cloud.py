import os
from dotenv import load_dotenv


load_dotenv()

PROFILE_NAME = os.environ["AWS_PROFILE_NAME"]
