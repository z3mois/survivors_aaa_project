FROM python:3.8

# Установка рабочей директории в контейнере
WORKDIR /app

# Копирование всех файлов из текущей директории в контейнер в директорию /app
COPY . /app

# Установка зависимостей
RUN pip install -r requirements.txt
RUN pip install torch==2.0.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# Запуск скрипта для построения базы данных
RUN python db_utils/create_and_fill_db.py

# Запуск сервера
CMD streamlit run serv.py