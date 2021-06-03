FROM python:3.7-slim-stretch
COPY . .
RUN pip install -r requirements.txt

ENV PATH_TO_MODEL="model_output_baseline.pkl"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]