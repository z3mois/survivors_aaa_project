import sqlite3
import pandas as pd


def create_connection(db_path):
    # create connection
    try:
        cnx = sqlite3.connect(db_path)
        return cnx
    except:
        print('Не получилось установить соединение с базой данных!')


# p2 = pd.read_sql('select City from price2', cnx)

def read_column_by_city(db_path, table_name, column_name, city):
    # избавляемся от регистра
    cnx = create_connection(db_path)
    city = city.lower()
    query = f"select {column_name} from {table_name} where City = '{city}'"

    df = pd.read_sql(query, cnx)
    return df[column_name].values

