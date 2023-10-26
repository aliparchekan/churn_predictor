import os
import logging
import churn_library_solution as cls
from churn_library import import_data

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

@pytest.fixture(scope= "module")
def raw_df():
	"""
	Return the raw data frame
	"""
	raw_df = import_data('./data/bank_data.csv')

	return raw_df

def test_import(import_data):
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
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):
	'''
	test perform eda function
	'''
	try:
		perform_eda(raw_df)
		files = ["age_histogram.png", "churn_histogram.png", "correlation.png", "marital_status_plot.png", "transaction_count_histogram.png"]
		for file in files:
			with open(file, 'r'):
				logging.info("Successfully opened %s", file)
	except FileNotFoundError:
		logging.error("ERROR: %s is not created", file)


def test_encoder_helper(encoder_helper):
	'''
	test encoder helper
	'''


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	pass








