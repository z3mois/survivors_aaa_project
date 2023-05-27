# параметры модели
MODEL_PARAMETERS = {
    "embedding_dim": 128,
    "rnn_hidden_dim": 512,
    "hidden_layers": 1,
    "fc1_output": 256,
    "fc2_output": 1,
    "rnn_type": 'GRU',
    "bidir": False,
    "dropout": 0,
    "vac_vocab_size": 19128,
    "res_vocab_size": 9425,
}

# указываем имя базы данных
DB_PATH = '.\\data\\cv2vac_project_db'

# указываем имя таблицы в базе данных
TABLE_NAME = 'avito_cv2vac'

# путь к исходному датафрейму
PATH_TO_DF = '.\\data\\avito_cv2vac_with_ranks_clear.pq'

# можно залить только часть
COLUMNS_TO_INSERT = []