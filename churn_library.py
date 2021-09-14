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
from sklearn.preprocessing import normalize
import shap
import joblib
import pandas as pd
import numpy as np
import constants
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(r'./images/eda/churn_distribution.png', bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(
        r'./images/eda/customer_age_distribution.png',
        bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(
        r'./images/eda/martial_status_distribution.png',
        bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], kde=True, stat="density", linewidth=0)
    plt.savefig(
        r'./images/eda/total_transaction_distribution.png',
        bbox_inches='tight')

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(r'./images/eda/heatmap.png', bbox_inches='tight')
    return


def encoder_helper(df, category_lst, response=constants.response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for feature in category_lst:
        df[feature + "_" +
            response] = df[feature].map(df.groupby(feature).mean()[response])
    return df


def perform_feature_engineering(df, response=constants.response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = constants.cat_columns
    quant_columns = constants.quant_columns
    y = df[response]
    X = pd.DataFrame()

    # here we encode categorical variables
    df = encoder_helper(df, cat_columns, response)
    keep_cols = quant_columns + df.columns[-len(cat_columns):].to_list()
    X[keep_cols] = df[keep_cols]

    # 30% of data goes to test distribution by default
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=constants.test_size, random_state=constants.random_state)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    # Random Forest classification report stored as an image
    plt.figure('figure_rf', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'./images/results/rf_results.png', bbox_inches='tight')

    # Logistic Regression classification report stored as an image
    plt.figure('figure_lr', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {
             'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(r'./images/results/logistic_results.png', bbox_inches='tight')
    return


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Shap plot
    plt.figure(figsize=(10, 10))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth + '_shap.png', bbox_inches='tight')

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth + '.png', bbox_inches='tight')
    return


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=constants.random_state)
    lrc = LogisticRegression(
        random_state=constants.random_state,
        max_iter=constants.max_iter)

    param_grid = constants.param_grid

    cv_rfc = GridSearchCV(
        estimator=rfc,
        param_grid=param_grid,
        cv=constants.cv)
    # Random Forest training
    cv_rfc.fit(X_train, y_train)
    # Logistic Regression training
    lrc.fit(X_train, y_train)

    # Store trained models as pkl files
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Get preds for both models
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Get and store classification reports
    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf)

    # Get and store ROC curves
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=constants.alpha)
    lrc_disp = plot_roc_curve(
        lrc,
        X_test,
        y_test,
        ax=ax,
        alpha=constants.alpha)
    plt.savefig(r'./images/results/roc_curve_result.png', bbox_inches='tight')

    # Get and store feature importance plot
    rfc_model = joblib.load('./models/rfc_model.pkl')
    feature_importance_plot(
        rfc_model,
        X_test,
        './images/results/feature_importance')
    return


if __name__ == "__main__":
    # Just normal workflow here for testing purposes
    data_load = import_data('./data/bank_data.csv')
    perform_eda(data_load)
    data_encoded = encoder_helper(data_load, constants.cat_columns)
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        data_encoded)
    train_models(X_train, X_test, y_train, y_test)
