'''
Churn logging and tests solution

author: Oleg
date: september 2021
'''

#pylint: disable=wrong-import-positio
import os
import logging
import pandas as pd
import constants as c
import churn_library as cl
#pylint: enable=wrong-import-position

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data, pth):
    '''
    test data import - this example is completed for you to assist with the other test functions

    input:
        import_data: function to be tested from churn library
                    pth: a path to the csv

    output:
        data_frame: pandas dataframe
'''
    try:
        data_frame = import_data(pth)
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
        return data_frame
    except FileNotFoundError:
        logging.error("Testing import_eda: The file wasn't found")
    except AssertionError:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
    return pd.DataFrame()

def test_eda(perform_eda, data_frame):
    '''
    test perform eda function

input:
        perform_eda: function to be tested from churn library
        data_frame: pandas dataframe

output:
        None
    '''
    try:
        perform_eda(data_frame)
        assert os.path.isfile('./images/eda/churn_distribution.png')
        assert os.path.isfile('./images/eda/customer_age_distribution.png')
        assert os.path.isfile('./images/eda/martial_status_distribution.png')
        assert os.path.isfile(
            './images/eda/total_transaction_distribution.png')
        assert os.path.isfile('./images/eda/heatmap.png')
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError:
        logging.error(
            "Testing perform_eda: The dataframe doesn't have proper data or empty")
    except TypeError:
        logging.error(
            "Testing perform_eda: The parameter doesn't appear to be a dataframe")
    except AssertionError:
        logging.error(
            "Testing perform_eda: One or more EDA images are are not in place")

def test_encoder_helper(encoder_helper, data_frame, category_lst):
    '''
    test encoder helper

input:
        encoder_helper: function to be tested from churn library
        data_frame: pandas dataframe
        category_lst: list of columns that contain categorical features

output:
        data_frame: pandas dataframe with new columns
    '''
    try:
        data_frame = encoder_helper(data_frame, category_lst)
        assert len(category_lst) > 0
        logging.info("Testing encoder_helper: SUCCESS")
        return data_frame
    except KeyError:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have proper data or empty")
    except TypeError:
        logging.error(
            "Testing encoder_helper: Function parameters have wrong type")
    except AssertionError:
        logging.error("Testing encoder_helper: Category list is empty")
    return pd.DataFrame()

def test_perform_feature_engineering(perform_feature_engineering, data_frame):
    '''
    test perform_feature_engineering

input:
          perform_feature_engineering: function to be tested from churn library
          data_frame: pandas dataframe

output:
          x_data_train: X training data
          x_data_test: X testing data
          y_data_train: y training data
          y_data_test: y testing data
    '''
    try:
        x_data_train, x_data_test, y_data_train, y_data_test = perform_feature_engineering(
            data_frame)
        assert len(x_data_train) > 0
        assert len(x_data_test) > 0
        assert len(y_data_train) > 0
        assert len(y_data_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
        return x_data_train, x_data_test, y_data_train, y_data_test
    except KeyError:
        logging.error(
            "Testing perform_feature_engineering: The dataframe doesn't have proper data or empty")
    except TypeError:
        logging.error(
            "Testing perform_feature_engineering: Function parameters have wrong type")
    except AssertionError:
        logging.error(
            "Testing perform_feature_engineering: Train and/or test set are empty")
    return 0, 0, 0, 0

def test_train_models(
        train_models,
        x_data_train,
        x_data_test,
        y_data_train,
        y_data_test):
    '''
    test train_models

input:
          train_models: function to be tested from churn library
          x_data_train: X training data
          x_data_test: X testing data
          y_data_train: y training data
          y_data_test: y testing data
output:
          None
    '''
    try:
        assert len(x_data_train) > 0
        assert len(x_data_test) > 0
        assert len(y_data_train) > 0
        assert len(y_data_test) > 0
    except AssertionError:
        logging.error("Testing train_models: Train and/or test set are empty")
        return
    except TypeError:
        logging.error(
            "Testing train_models: One or more parameters have wrong type")
        return

    try:
        train_models(x_data_train, x_data_test, y_data_train, y_data_test)
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('./images/results/rf_results.png')
        assert os.path.isfile('./images/results/logistic_results.png')
        assert os.path.isfile('./images/results/feature_importance.png')
        assert os.path.isfile('./images/results/feature_importance_shap.png')
        logging.info("Testing train_models: SUCCESS")
    except TypeError:
        logging.error(
            "Testing train_models: One or more parameters have wrong type")
    except AssertionError:
        logging.error(
            "Testing train_models: Some models or result images are missed")

if __name__ == "__main__":
    # functions test set
    data_load = test_import(cl.import_data, "./data/bank_data2.csv")  # ERROR
    data_load = test_import(cl.import_data, "./data/bank_data.csv")  # SUCCESS
    test_eda(cl.perform_eda, data_load)  # SUCCESS
    test_eda(cl.perform_eda, 5)  # ERROR
    test_eda(cl.perform_eda, pd.DataFrame())  # ERROR
    data_encoded = test_encoder_helper(
        cl.encoder_helper, data_load, [])  # ERROR
    data_encoded = test_encoder_helper(
        cl.encoder_helper, 5, c.CAT_COLUMNS)  # ERROR
    data_encoded = test_encoder_helper(
        cl.encoder_helper,
        pd.DataFrame(),
        c.CAT_COLUMNS)  # ERROR
    data_encoded = test_encoder_helper(
        cl.encoder_helper, data_load, c.CAT_COLUMNS)  # SUCCESS
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cl.perform_feature_engineering, 5)  # ERROR
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cl.perform_feature_engineering, pd.DataFrame())  # ERROR
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cl.perform_feature_engineering, data_encoded)  # SUCCESS
    test_train_models(cl.train_models, 0, 5, 'string', y_test)  # ERROR
    test_train_models(
        cl.train_models,
        x_train,
        x_test,
        y_train,
        y_test)  # SUCCESS
