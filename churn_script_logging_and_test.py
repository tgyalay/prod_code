'''
This script contains unit tests (pytest), a main() function for running
the analysis, and also runs the analysis if the script is called
directly
Created on Wednesday June 1 by Tamas Gyalay
'''
import os
import logging
from time import time
from glob import glob
from contextlib import contextmanager

import pytest
import pandas as pd

from churn_library import ChurnAnaysis, import_data, encoder_helper
import constants

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.DEBUG,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@contextmanager
def output_file_checker(required_files):
    '''
    This function can be used as a context to check if the required
    files were created in the root directory.

    input:
            set: a set containing the full filepaths from the current
            working directory.
    '''
    begin_time = time()
    try:
        yield
    finally:
        found_files = set(glob('./**/*', recursive=True))
        try:
            # examine only the files we are looking for
            # test that every required file is present in eda_files
            found_files &= required_files
            assert required_files <= found_files
            logging.info('All required files are present')
        except AssertionError as err:
            logging.error('Some or all files were not produced!')
            raise err

        try:
            creation_times_list = [os.path.getctime(file)
                                   for file in found_files]
            assert all(element - begin_time > 0
                       for element in creation_times_list)
            logging.info('All required files were updated')
        except AssertionError as err:
            logging.error('Some or all of the files were not updated!')
            raise err


@pytest.fixture(name="df_imported")
def fixture_df_imported():
    '''
    This fixture creates an instance of the churnalysis with the data
    already imported and the Churn column initialized.
    '''
    return ChurnAnaysis(constants.data_path)


def test_import():
    '''
    test data import - this example is completed for you to assist with
    the other test functions
    '''
    try:
        dataframe = import_data(constants.data_path)
        logging.info("SUCCESS: testing import_data")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert dataframe.shape[0] > 0
        assert dataframe.shape[1] > 0
        logging.info(
            "There is a non-zero number of rows and columns in the DataFrame")
    except AssertionError as err:
        logging.error(
            "testing import: The file doesn't appear to have rows and columns")
        raise err


def test_eda(df_imported):
    '''
    test perform eda function
    '''

    eda_required_files = constants.required_eda_ouputs

    with output_file_checker(eda_required_files):
        try:
            df_imported.perform_eda()
            logging.info('SUCCESS: No errors drawing plots')
        except KeyError as err:
            logging.error(
                "For EDA to complete the columns need to include: 'Churn',"\
                "'Customer Age', 'Marital_Status', 'Total_Trans_Ct'")
            raise err


def test_encoder_helper(df_imported):
    '''
        test encoder helper
        '''
    old_df = df_imported.df.copy()

    gender_lst = []
    gender_groups = old_df.groupby('Gender').mean()['Churn']

    for val in old_df['Gender']:
        gender_lst.append(gender_groups.loc[val])

    old_df['Gender_Churn'] = gender_lst

    new_df = encoder_helper(df_imported.df,
                            category_lst=['Gender'],
                            response="Churn")

    try:
        assert (old_df['Gender_Churn'] == new_df['Gender_Churn']).all
        logging.info('Churn Proportion computed correctly')
    except AssertionError as err:
        logging.error('Churn Proportion was not computed correctly')
        raise err


def test_perform_feature_engineering(df_imported):
    '''
        test perform_feature_engineering
        '''

    response = 'Churn'
    keep_cols = (constants.non_cat_keep_cols
                 + [category + '_' + response
                    for category in constants.cat_columns
                    ]
                 )

    df_imported.perform_feature_engineering()

    try:
        assert (df_imported.X_train.columns == keep_cols).all
        logging.info('The correct features are in the training set')
    except AssertionError as err:
        logging.error('The features in the training set are not correct!')
        raise err

    try:
        assert isinstance(df_imported.y_train, pd.Series)
        logging.info('There is correctly one response variable for the model')
    except AssertionError as err:
        logging.error('The response variable has more than one value!')
        raise err


def test_train_models(df_imported):
    '''
        test train_models
        '''

    train_output_files = constants.required_training_outputs

    df_imported.perform_feature_engineering()

    with output_file_checker(train_output_files):
        df_imported.train_models()


def main(pth):
    '''
    main function that can be called with a reference to a specific
    filepath in the arugment if this script is imported.

    input:
            pth: a path to the csv

    output:
            none
    '''
    churn_obj = ChurnAnaysis(pth)
    churn_obj.perform_eda()
    churn_obj.perform_feature_engineering()
    churn_obj.train_models()


if __name__ == "__main__":
    main(constants.data_path)
