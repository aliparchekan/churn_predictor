'''
Module to test and log the functionality of the churn_library.py

Author: Ali

Date: October 2023
'''

import os
import logging
import pytest
from churn_library import import_data, perform_eda, perform_feature_engineering, encoder_helper, train_models
import joblib

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s', force = True)


@pytest.fixture(scope="module")
def raw_df():
    """
    Return the raw data frame
    """
    df = import_data('./data/bank_data.csv')

    return df


@pytest.fixture(scope="module")
def encoded_df(raw_df):
    perform_eda(raw_df)
    encoded_df = encoder_helper(raw_df, ["Gender",
                                         'Education_Level',
                                         'Marital_Status',
                                         'Income_Category',
                                         'Card_Category'],
                                'Churn')

    return encoded_df


@pytest.fixture(scope="module")
def featured_df(encoded_df):
    featured_df = perform_feature_engineering(encoded_df, 'Churn')
    return featured_df


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(raw_df):
    '''
    test perform eda function
    '''
    try:
        perform_eda(raw_df)
        files = [
            "age_histogram.png",
            "churn_histogram.png",
            "correlation.png",
            "marital_status_plot.png",
            "transaction_count_histogram.png"]
        for file in files:
            with open('./images/' + file, 'r'):
                logging.info("Successfully opened %s", file)
    except FileNotFoundError:
        logging.error("ERROR: Missing images")


def test_encoder_helper(raw_df):
    '''
    test encoder helper
    '''
    encoded_df = encoder_helper(raw_df, ["Gender",
                                         'Education_Level',
                                         'Marital_Status',
                                         'Income_Category',
                                         'Card_Category'],
                                'Churn')

    try:
        for column in [
            "Gender",
            'Education_Level',
            'Marital_Status',
            'Income_Category',
                'Card_Category']:
            assert column + "_Churn" in encoded_df
        logging.info("Successfully added all the columns")
    except AssertionError:
        logging.error("ERROR: Missing columns")


def test_perform_feature_engineering(encoded_df):
    '''
    test perform_feature_engineering
    '''
    try:
        FE_DF = perform_feature_engineering(encoded_df, 'Churn')
        x_train, x_test, y_train, y_test = FE_DF[0], FE_DF[1], FE_DF[2], FE_DF[3]

        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Successfully divided data into train and test")
    except AssertionError:
        logging.error("ERROR: Incompatible size of data")


def test_train_models(featured_df):
    '''
    test train_models
    '''
    try:
        train_models(
            featured_df[0],
            featured_df[1],
            featured_df[2],
            featured_df[3])
        joblib.load("models/rfc_model.pkl")
        joblib.load("models/logistic_model.pkl")
        logging.info("Successfully loaded models")
    except FileNotFoundError:
        logging.error("ERROR: missing model(s)")

    try:
        for name in [
            "Random Forest Train",
            "Random Forest Test",
            "Logistic Regression Train",
                "Logistic Regression Test"]:
            with open('./images/' + name.replace(' ', '_') + '.png', 'r'):
                logging.info("Successfully opened %s", name)

    except FileNotFoundError:
        logging.error("ERROR: missing an image")


if __name__ == "__main__":
    pass
