from ast import Assert
import os
import logging
from datetime import datetime
from glob import glob

import pandas as pd

from churn_library import (import_data, perform_eda, 
    perform_feature_engineering, encoder_helper, 
    classification_report_image, feature_importance_plot, train_models)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

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
    df = import_data('./data/bank_data.csv')
    start_time = datetime.now()

    try:
        perform_eda(df)
        logging.info('SUCCESS: No errors drawing plots')
    except KeyError as err:
        logging.error("For EDA to complete the columns need to include 'Churn'\
            , 'Customer Age', 'Marital_Status', 'Total_Trans_Ct'")
        raise err

    required_files = set([
        './images/eda/churn.png',
        './images/eda/customer_age.png',
        './images/eda/marital_status.png',
        './images/eda/transaction_distribution.png',
        './images/eda/correlation_heatmap.png',                        
        ])
    eda_files = set(glob('./images/eda/*'))

    try:
        #examine only the files we are looking for
        #test that every required file is present in eda_files
        eda_files &= required_files
        assert required_files <= eda_files
        logging.info('All required files are present')
    except AssertionError as err:
        logging.error('Some or all files were not produced!')
        raise err

    try:
        creation_times_list = [os.path.getctime(file) for file in eda_files]
        assert all(element-start_time > 0 for element in creation_times_list)
        logging.info('All present files were updated')
    except AssertionError as err:
        logging.error('Some or all of the files were not updated!')
        raise err


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








