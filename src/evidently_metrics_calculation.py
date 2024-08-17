"""Script to simulate model monitoring with Evidently in the test set and
store the metrics in a PostgreSQL database."""
# pylint: disable=logging-fstring-interpolation
import logging
import pickle

import pandas as pd
from tqdm import tqdm
import psycopg
from evidently.metrics import (ClassificationClassBalance,
                               ClassificationConfusionMatrix,
                               ClassificationDummyMetric,
                               ClassificationQualityByClass,
                               ClassificationQualityMetric,
                               ConflictPredictionMetric, ConflictTargetMetric)
from evidently.report import Report
from prefect import flow, task


@task
def prep_db(conn_str: str, table_str: str, create_table_statement: str):
    """Prepare the database for the metrics.

    Args:
        conn_str (str): Connection string to create the database.
        table_str (str): Connection string to connect to the database.
        create_table_statement (str): SQL statement to create the table.
    """
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
def calculate_metrics_postgresql(cur: psycopg.cursor,
                                 i: int, model: object, report: object,
                                 reference_data: pd.DataFrame,
                                 test_data: pd.DataFrame,
                                 batch_size: int = 50):
    """Calculate the metrics for raw data and store them in the database.

    Args:
        cur (psycopg.cursor): Cursor to the database.
        i (int): Index to start the batch.
        model (object): Model to predict the data.
        report (object): Report object to calculate the metrics.
        reference_data (pd.DataFrame): Reference data to calculate the metrics.
        test_data (pd.DataFrame): Raw data to calculate the metrics.
        batch_size (int): Size of the batch to calculate the metrics.
    """
    current_data = test_data.iloc[i:i + batch_size]

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

    batch_number = int(i / batch_size)

    cur.execute(
        """insert into heart_metrics(batch_number, precision_,
        recall, f1, accuracy) values
        (%s, %s, %s, %s, %s)""",
        (batch_number, precision, recall, f1, accuracy)
    )


@flow
def batch_monitoring_backfill(table_str: str, test_data: pd.DataFrame,
                              model: object, report: object,
                              reference_data: pd.DataFrame,
                              batch_size: int = 50):
    """Flow to calculate the metrics for the test data and store them in the
    database. The metrics are calculated for every batch of 10 rows."""
    conn = psycopg.connect(table_str, autocommit=True)
    cur = conn.cursor()
    for i in tqdm(range(0, len(test_data), batch_size),
                  desc="Calculating metrics"):
        calculate_metrics_postgresql(cur, i, model, report,
                                     reference_data, test_data, batch_size)
        logging.info("data sent")


def main():
    """Main function to run the flow."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

    batch_size = 500
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

    reference_data = pd.read_parquet(
        'data/processed/heart_train_cleaned.parquet')

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

    prep_db(conn_str, table_str, create_table_statement)
    batch_monitoring_backfill(table_str, test_data,
                              model, report, reference_data, batch_size)


if __name__ == "__main__":
    main()
