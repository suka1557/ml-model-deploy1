import os
import sys
PROJECT_ROOT = os.path.abspath("./")
sys.path.append(PROJECT_ROOT)

from dotenv import load_dotenv
if os.path.exists(os.path.join(PROJECT_ROOT, 'secrets.env') ):
    load_dotenv(os.path.join(PROJECT_ROOT, 'secrets.env'))

def load_aws_credentials_into_memory():
    #LOAD CREDENTIALS
    os.environ['AWS_ACCESS_KEY_ID'] = os.getenv("AWS_ACCESS_KEY_ID")
    os.environ['AWS_SECRET_ACCESS_KEY'] = os.getenv("AWS_SECRET_ACCESS_KEY")
    os.environ['region_name'] = os.getenv("region_name")
    os.environ['MYSQL_URI'] = os.getenv("MYSQL_URI")