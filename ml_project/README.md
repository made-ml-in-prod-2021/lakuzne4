ml_project
==============================

В данном проекте разработан пайплайн для обучения моделей на датасете 
Heart diseases USI (https://www.kaggle.com/ronitf/heart-disease-uci)

Основные части пайплайна (что и для чего сделано):
1) Модуль чтения конфигурационных файлов - для удобного изменения входных данных и параметров обучения/валидации
2) Модуль обработки данных - для чтения и разделения данных на train/val/test
3) Модуль преобразования данных - для предобработки данных с помощью трансформеров (в том числе и кастомных)
4) Модуль обучения модели - для обучения модели и её сохранения в папку models для будущего использования
5) Функции валидации - для подсчёта метрик из конфигурационного файла и сохранения результатов в json-формате для сравнения с другими моделями.


Для обучения модели следует использовать команду train с заполненным файлом конфигурации:
    
    python ml_project/train_pipeline.py train --config_file configs/base_config_log_reg.yml



Для предсказания с помощью уже обученной модели следует использовать команду predict с трёмя обязательными опциями:

    python ml_project/train_pipeline.py predict --data_path data/interim/sample_for_prediction.csv --model_path models/model_output_GB.pkl --output_path data/predicted/predicted.csv


Структура проекта
------------

    ├── README.md          <- Главный README файл для пользователей.
    ├── train_pipeline.py  <- Файл с верхнеуровневым кодом для пайплайна обучения
    ├── data
    │   ├── interim        <- Папка для возможных промежуточных данных
    │   ├── processed      <- Окончательный набор данных для моделирования
    │   └── raw            <- Исходный набор данных
    |
    ├── logs               <- Для логирования    
    │
    ├── models             <- Обученные и сохранённые модели, а также их предсказания
    │
    ├── notebooks          <- Jupyter notebooks для EDA задач 
    │
    ├── reports            <- Отчёты по прогонам моделей
    │   └── figures        <- Подпапка для сгенерированных графиков для отчётов
    │
    ├── requirements.txt   <- Файл requirements.txt для среды выполнения
    │
    └── source_code        <- Папка с исходным кодом проекта
        ├── __init__.py    <- Чтобы source_code был Python модулем.
        │
        ├── data           <- Скрипты для загрузки данных
        │   └── make_dataset.py
        |   
        ├── entities       <- Скрипты для обработки параметров конфигурации
        |   | 
        │   └── input_objects.py  <- объекты моделей и скореров для подтягивания их через config
        |   └── parameters.py <- параметры файлов конфигурации
        │
        ├── features       <- Скрипты для предобработки данных
        │   └── build_features.py
        │
        ├── models         <- Скрипты для тренировки моделей
            │                 
            ├── predict_model.py
            └── train_model.py
--------

Самооценка:
1) ветка homework1 - 1 балл
2) описание проекта - 2 балла
3) выполнено EDA - 2 балла
4) проект имеет модульную структуру - 2 балла
5) в проекте используются логгеры - 2 балла
6) написаны тесты на все модули и end2end тест - 3 балла
7) для тестов генерируются синтетичесие данные, как с помощью библиотеки faker, так и с помощью самописного генератора - 3 балла
8) обучение конфигурируется с помощью yaml-файла - 3 балла
9) для конфигурации используются датаклассы - 3 балла
10) написан кастомный трансформер - 2 балла
11) обучено несколько моделей и записаны результаты - 3 балла
12) написана cli-функция predict - 3 балла
13) проведена самооценка - 1 балл

Итого: 30 баллов
 