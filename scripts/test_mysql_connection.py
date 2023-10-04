import mysql.connector
from dotenv import load_dotenv
import os

#Read env file
load_dotenv('./secrets.env')

#Get db details
host=os.getenv("MYSQL_HOST")
user=os.getenv("MYSQL_USER")
password=os.getenv("MYSQL_PASSWORD")
database=os.getenv("MYSQL_DATABASE")
port=os.getenv("MYSQL_PORT")


try:
    connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port #Default port used is 3306 if not given
    )

    print("Connection Established Successfully")
except mysql.connector.Error as err:
    print(f"Error: {err}")