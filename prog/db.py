import pymysql, sqlalchemy
import pandas as pd



class Database():
    def __init__(self, db_con):
        self.py_conn, self.alchemy_conn = self.connect_db(db_con["host"], db_con["port"], db_con["user"], db_con["password"], db_con["database"])
        self.cursor = self.py_conn.cursor()


    def connect_db(self, host, port, user, password, database):
        py_conn = pymysql.connect(host = host, port = port, user = user, password = password, database = database)

        con_info = f'mysql+pymysql://{user}:{password}@{host}:{port}/{database}'
        alchemy_conn = sqlalchemy.create_engine(con_info)

        return py_conn, alchemy_conn
    

    def get_data(self, table, start_time = None, end_time = None):
        if start_time:
            query = f'SELECT * FROM {table} WHERE modify_time BETWEEN "{start_time}" AND "{end_time}"'
        else:
            query = f'SELECT * FROM {table}'
            
        df = pd.read_sql(query, self.alchemy_conn)

        return df
    

    def save_progress(self, percent):
        query = f'update a_common_single_predict_progress set percent = {percent}'
        self.cursor.execute(query)
        self.py_conn.commit()