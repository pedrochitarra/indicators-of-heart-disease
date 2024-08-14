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
from evidently import ColumnMapping
from evidently.metrics import (ColumnDriftMetric, DatasetDriftMetric,
                               DatasetMissingValuesMetric,
                               DatasetSummaryMetric,
                               ClassificationPreset)

# TODO: Use Evidendly to track Classification metrics to generate the reports.

# TODO: Use Grafana Cloud to visualize the metrics and dashboards. Learn how to
# use Grafana Cloud to visualize the metrics and dashboards.

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
    sample_number integer,
    prediction_drift float,
    num_drifted_columns integer,
    share_missing_values float
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

column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    target=None
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
    DatasetSummaryMetric(),
    ClassificationPreset()
])

conn_str = "host=localhost port=5432 user=postgres password=example"
table_str = ("host=localhost port=5432 dbname=test "
             "user=postgres password=example")


@task
def prep_db():
    """Prepare the database for the metrics."""
    conn = psycopg.connect(conn_str, autocommit=True)
    res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
    if len(res.fetchall()) == 0:
        conn.execute("create database test;")
    conn_table = psycopg.connect(table_str)
    conn_table.execute(create_table_statement)


@task
def calculate_metrics_postgresql(curr, i):
    """Calculate the metrics for raw data and store them in the database."""
    current_data = test_data.iloc[i:i + 1]

    # current_data.fillna(0, inplace=True)
    current_data = current_data.drop(columns=["HadHeartAttack"])

    if current_data.isna().sum().sum() > 0:
        current_data.fillna(0, inplace=True)

    prediction = model.predict(current_data)
    current_data['prediction'] = prediction

    report.run(reference_data=reference_data, current_data=current_data,
               column_mapping=column_mapping)

    result = report.as_dict()

    prediction_drift = result['metrics'][0]['result']['drift_score']
    num_drifted_columns = result['metrics'][1]['result'][
        'number_of_drifted_columns']
    share_missing_values = result['metrics'][2]['result']['current'][
        'share_of_missing_values']

    curr.execute(
        """insert into dummy_metrics(sample_number, prediction_drift,
        num_drifted_columns, share_missing_values) values
        (%s, %s, %s, %s)""",
        (i, prediction_drift,
         num_drifted_columns, share_missing_values)
    )


@flow
def batch_monitoring_backfill():
    """Flow to calculate the metrics for the test data and store them in the
    database."""
    prep_db()
    last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
    conn = psycopg.connect(table_str, autocommit=True)
    cur = conn.cursor()
    for i in range(0, len(test_data)):
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

    batch_monitoring_backfill()
