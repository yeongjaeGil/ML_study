import pymysql
import pandas as pd
host_name = "localhost"
username = "rlfdudwo"
password = "rlfdudwo"
#database_name = "study_db"

db = pymysql.connect(
    host=host_name,  # DATABASE_HOST
    port=3306,
    user=username,  # DATABASE_USERNAME
    passwd=password,  # DATABASE_PASSWORD
    #db=database_name,  # DATABASE_NAME
    charset='utf8'
)

SQL = "SHOW TABLES"