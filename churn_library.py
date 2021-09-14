'''
Churn library project solution

author: Oleg
date: september 2021
'''

from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import constants as c
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            data_frame: pandas dataframe
    '''
    data_frame = pd.read_csv(pth)
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on data frame and save figures to images folder
    input:
            data_frame: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    data_frame['Churn'].hist()
    plt.savefig(r'./images/eda/churn_distribution.png', bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    data_frame['Customer_Age'].hist()
    plt.savefig(
        r'./images/eda/customer_age_distribution.png',
        bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        r'./images/eda/martial_status_distribution.png',
        bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    sns.histplot(
        data_frame['Total_Trans_Ct'],
        kde=True,
        stat="density",
        linewidth=0)
    plt.savefig(
        r'./images/eda/total_transaction_distribution.png',
        bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(r'./images/eda/heatmap.png', bbox_inches='tight')


def encoder_helper(data_frame, category_lst, response=c.RESPONSE):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            data_frame: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used
                      for naming variables or index y column]

    output:
            data_frame: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        data_frame[feature + "_" + response] = data_frame[feature].map(
            data_frame.groupby(feature).mean()[response])
    return data_frame


def perform_feature_engineering(data_frame, response=c.RESPONSE):
    '''
    input:
              data_frame: pandas dataframe
              response: string of response name [optional argument that could be used
                        for naming variables or index y column]

    output:
              x_data_train: X training data
              x_data_test: X testing data
              y_data_train: y training data
              y_data_test: y testing data
    '''
    cat_columns = c.CAT_COLUMNS
    quant_columns = c.QUANT_COLUMNS
    y_data = data_frame[response]
    x_data = pd.DataFrame()

    # here we encode categorical variables
    data_frame = encoder_helper(data_frame, cat_columns, response)
    keep_cols = quant_columns + \
        data_frame.columns[-len(cat_columns):].to_list()
    x_data[keep_cols] = data_frame[keep_cols]

    # 30% of data goes to test distribution by default
    x_data_train, x_data_test, y_data_train, y_data_test = train_test_split(
        x_data, y_data, test_size=c.TEST_SIZE, random_state=c.RANDOM_STATE)
    return x_data_train, x_data_test, y_data_train, y_data_test


def classification_report_image(x_y_data):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            x_y_data[0]: training response values
            x_y_data[1]:  test response values
            x_y_data[2]: training predictions from logistic regression
            x_y_data[3]: training predictions from random forest
            x_y_data[4]: test predictions from logistic regression
            x_y_data[5]: test predictions from random forest

    output:
             None
    '''
    # Random Forest classification report stored as an image
    plt.figure('figure_rf', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(x_y_data[1], x_y_data[5])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(x_y_data[0], x_y_data[3])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'./images/results/rf_results.png', bbox_inches='tight')

    # Logistic Regression classification report stored as an image
    plt.figure('figure_lr', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(x_y_data[0], x_y_data[2])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(x_y_data[1], x_y_data[4])), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'./images/results/logistic_results.png', bbox_inches='tight')


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Shap plot
    plt.figure(figsize=(10, 10))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_data)
    shap.summary_plot(shap_values, x_data, plot_type="bar", show=False)
    plt.savefig(output_pth + '_shap.png', bbox_inches='tight')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + '.png', bbox_inches='tight')


def train_models(x_data_train, x_data_test, y_data_train, y_data_test):
    '''
    train, store model results: images + scores, and store models
    input:
              x__data_train: X training data
              X_data_test: X testing data
              y_data_train: y training data
              y_data_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=c.RANDOM_STATE)
    lrc = LogisticRegression(
        random_state=c.RANDOM_STATE,
        max_iter=c.MAX_INTER)

    param_grid = c.PARAM_GRID

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=c.CV)
    # Random Forest training
    cv_rfc.fit(x_data_train, y_data_train)
    # Logistic Regression training
    lrc.fit(x_data_train, y_data_train)

    # Store trained models as pkl files
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Get preds for both models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_data_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_data_test)
    y_train_preds_lr = lrc.predict(x_data_train)
    y_test_preds_lr = lrc.predict(x_data_test)

    # Get and store classification reports
    x_y_data = [
        y_data_train,
        y_data_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf]
    classification_report_image(x_y_data)

    # Get and store ROC curves
    plt.figure(figsize=(15, 8))
    axes = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_data_test,
        y_data_test,
        ax=axes,
        alpha=c.ALPHA)
    plot_roc_curve(
        lrc,
        x_data_test,
        y_data_test,
        ax=axes,
        alpha=c.ALPHA)
    plt.savefig(r'./images/results/roc_curve_result.png', bbox_inches='tight')

    # Get and store feature importance plot
    rfc_model = joblib.load('./models/rfc_model.pkl')
    feature_importance_plot(
        rfc_model,
        x_data_test,
        './images/results/feature_importance')


if __name__ == "__main__":
    # Just normal workflow here for testing purposes
    data_load = import_data('./data/bank_data.csv')
    perform_eda(data_load)
    data_encoded = encoder_helper(data_load, c.CAT_COLUMNS)
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data_encoded)
    train_models(x_train, x_test, y_train, y_test)
