## Airflow ML DAGS

Для запуска в начале прописать FERNET_KEY:
  
    export FERNET_KEY=$(python -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")

Далее запустить Docker-compose:

    docker-compose up --build
    
Что сделано:

    1) Локально поднят контейнер airflow. Реализован DAG, генерирующий данные для обучения модели. - 5 баллов
    2) Реализован dag, который обучает модель еженедельно, используя данные за текущий день. - 10 баллов
    3) Реализован dag, который использует модель ежедневно. - 5 баллов.
    4) Все dags реализованы с помощью Docker Operator. - 10 баллов
    5) Сенсоры на готовность данных. - 3 балла
    5) Самооценка - 1 балл.

Итого: 34 балла.
