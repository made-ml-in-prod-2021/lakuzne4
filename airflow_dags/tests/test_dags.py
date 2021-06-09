import pytest
from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    return DagBag(dag_folder="dags/",
                  include_examples=False)
                  
                  
def test_downloading(dag_bag: DagBag):
    dag = dag_bag.get_dag(dag_id='download_data')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 3
    
