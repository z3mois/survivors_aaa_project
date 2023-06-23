import sqlite3
import sys
import os
import pandas as pd
sys.path.append('data')
print(os.path.abspath(os.curdir))
print(sys.path)
from constants import PATH_TO_DF, DB_PATH, TABLE_NAME, COLUMNS_TO_INSERT




def create_connection(db_path):
    # create connection
    try:
        cnx = sqlite3.connect(db_path)
        return cnx
    except:
        print('Не получилось установить соединение с базой данных!')


def write_df_to_sql(df_path, cols_to_insert, table_name, cnx):
    """
        create database base on df 
    """
    df = pd.read_parquet(df_path)
    if len(cols_to_insert) > 0:
        df = df[cols_to_insert]

    # перед заливкой избавимся от чувствительности к регистру при фильтрации
    df['City'] = df['City'].str.lower()

    df.to_sql(name=table_name, con=cnx, if_exists='replace')


if __name__ == "__main__":
    print('Создаю базу данных и заливаю таблицу...')
    connection = create_connection(DB_PATH)

    write_df_to_sql(PATH_TO_DF, COLUMNS_TO_INSERT, TABLE_NAME, connection)

