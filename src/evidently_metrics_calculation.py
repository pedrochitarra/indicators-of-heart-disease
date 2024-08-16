"""Script to simulate model monitoring with Evidently in the test set and
store the metrics in a PostgreSQL database."""
# pylint: disable=logging-fstring-interpolation
import pickle
import random
import logging
import datetime
import time

import pandas as pd
import psycopg
import yaml
from prefect import task, flow
from evidently.report import Report
from evidently.metrics import ConflictTargetMetric
from evidently.metrics import ConflictPredictionMetric
from evidently.metrics import ClassificationQualityMetric
from evidently.metrics import ClassificationClassBalance
from evidently.metrics import ClassificationConfusionMatrix
from evidently.metrics import ClassificationQualityByClass
from evidently.metrics import ClassificationDummyMetric

# TODO: Use Grafana Cloud to visualize the metrics and dashboards. Learn how to
# use Grafana Cloud to visualize the metrics and dashboards.

# TODO: Organize this code in a more modular way!

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists heart_metrics;
create table heart_metrics(
    batch_number integer,
    precision_ float,
    recall float,
    f1 float,
    accuracy float
)
"""

test_data = pd.read_parquet('data/interim/heart_test.parquet')
with open('model.pkl', 'rb') as f_in:
    model = pickle.load(f_in)

reference_data = pd.read_parquet('data/processed/heart_train_cleaned.parquet')

# Load constants/categorical_features from params.yaml
with open("params.yaml", encoding="utf-8") as file:
    dvc_params = yaml.safe_load(file)

cat_features = dvc_params["categorical_features"]
num_features = list(set(reference_data.columns) -
                    set(cat_features) - {'HadHeartAttack'})

report = Report(metrics=[
    ClassificationQualityMetric(),
    ClassificationClassBalance(),
    ConflictTargetMetric(),
    ConflictPredictionMetric(),
    ClassificationConfusionMatrix(),
    ClassificationQualityByClass(),
    ClassificationDummyMetric(),
])

conn_str = "host=localhost port=5432 user=postgres password=example"
table_str = ("host=localhost port=5432 dbname=test "
             "user=postgres password=example")


@task
def prep_db():
    """Prepare the database for the metrics."""
    conn = psycopg.connect(conn_str)
    cur = conn.cursor()
    res = cur.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    conn.commit()
    if len(res.fetchall()) == 0:
        cur.execute("create database test;")
        conn.commit()
    conn_table = psycopg.connect(table_str)
    cur_table = conn_table.cursor()
    cur_table.execute(create_table_statement)
    conn_table.commit()


@task
def calculate_metrics_postgresql(cur, i):
    """Calculate the metrics for raw data and store them in the database."""
    current_data = test_data.iloc[i:i + 10]

    if current_data.isna().sum().sum() > 0:
        current_data.fillna(0, inplace=True)

    # current_data.fillna(0, inplace=True)
    real_value = current_data['HadHeartAttack'].values
    current_data = current_data.drop(columns=["HadHeartAttack"])

    prediction = model.predict(current_data)
    current_data['prediction'] = prediction
    current_data['target'] = real_value

    current_data['prediction'] = current_data['prediction'].replace(
        {"No": 0, "Yes": 1})
    current_data['target'] = current_data['target'].replace(
        {"No": 0, "Yes": 1})

    report.run(reference_data=reference_data, current_data=current_data)

    result = report.as_dict()

    precision = result['metrics'][0]['result']['current']['precision']
    recall = result['metrics'][0]['result']['current']['recall']
    f1 = result['metrics'][0]['result']['current']['f1']
    accuracy = result['metrics'][0]['result']['current']['accuracy']

    batch_number = int(i / 10)

    cur.execute(
        """insert into heart_metrics(batch_number, precision_,
        recall, f1, accuracy) values
        (%s, %s, %s, %s, %s)""",
        (batch_number, precision, recall, f1, accuracy)
    )


@flow
def batch_monitoring_backfill():
    """Flow to calculate the metrics for the test data and store them in the
    database."""
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    conn = psycopg.connect(table_str, autocommit=True)
    cur = conn.cursor()
    for i in range(0, len(test_data), 10):
        calculate_metrics_postgresql(cur, i)
        new_send = datetime.datetime.now()
        seconds_elapsed = (new_send - last_send).total_seconds()
        if seconds_elapsed < SEND_TIMEOUT:
            time.sleep(SEND_TIMEOUT - seconds_elapsed)
        while last_send < new_send:
            last_send = last_send + datetime.timedelta(seconds=2)
        logging.info("data sent")


if __name__ == '__main__':
    reference_data_preds = model.predict(reference_data.drop(columns=[
        "HadHeartAttack"]))
    reference_data['prediction'] = reference_data_preds
    # Rename the target column to 'target' to generate the report
    reference_data = reference_data.rename(
        columns={"HadHeartAttack": "target"})
    reference_data['prediction'] = reference_data['prediction'].replace(
        {"No": 0, "Yes": 1})
    reference_data['target'] = reference_data['target'].replace(
        {"No": 0, "Yes": 1})

    batch_monitoring_backfill()
