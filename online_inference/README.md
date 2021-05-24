online_inference
==============================

Описание REST API:
```/``` - GET, root endpoint
```/predict``` - POST, сделать предсказание для списка строк данных
```docs``` - документация

Сделано:
   1) REST - сервис app.py
   2) тест для predict - tests/test_app.py:test_predict
   3) Скрипт делающий запросы к REST-сервису: make_requests.py
   4) Валидация данных - src/data_validation.py
   5) dockerfile - Docker

