FROM airflow-ml-base:latest

COPY train.py /train.py

ENTRYPOINT ["python", "train.py", "--input_dir", ".", "--output_dir", "."]