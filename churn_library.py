'''
Module to load, extract features, train, and test to identify churning customers

Author: Ali

Date: October 2023
'''


# import libraries
import os
import joblib
from scikitplot.metrics import plot_roc
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import RocCurveDisplay
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

os.environ['QT_QPA_PLATFORM'] = 'offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig('./images/churn_histogram.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig('./images/age_histogram.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/marital_status_plot.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/transaction_count_histogram.png')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(
        df.corr(
            numeric_only=True),
        annot=False,
        cmap='Dark2_r',
        linewidths=2)
    plt.savefig('./images/correlation.png')
    plt.close()


def encoder_helper(df, category_lst, response):
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

    for category in category_lst:
        groups = df.groupby(category).mean(numeric_only=True)[response]
        lst = []
        for val in df[category]:
            lst.append(groups.loc[val])

        df[category + "_" + response] = lst

    return df


def perform_feature_engineering(df, response):
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
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X = pd.DataFrame()
    y = df[response]
    X[keep_cols] = df[keep_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

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
    result_dict = {
        "Random Forest Train": (y_train, y_train_preds_rf),
        "Random Forest Test": (y_test, y_test_preds_rf),
        "Logistic Regression Train": (y_train, y_train_preds_lr),
        "Logistic Regression Test": (y_test, y_test_preds_lr)
    }

    for evaluation, data in result_dict.items():
        fig = plt.figure(figsize=(5,5))
        plt.text(
            0.01, 1.25, evaluation, {
                'fontsize': 10}, fontproperties='monospace', transform=fig.transFigure)
        plt.text(
            0.01, 0.05, str(
                classification_report(
                    data[0], data[1])), {
                'fontsize': 10}, fontproperties='monospace', transform=fig.transFigure)
        plt.axis('off')
        plt.savefig('./images/' + evaluation.replace(' ', '_') + '.png')


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
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 12))
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(output_pth)


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
    rfc = RandomForestClassifier(random_state=42)

    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    lrc_plot = RocCurveDisplay.from_estimator(lrc, X_test, y_test)


    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = RocCurveDisplay.from_estimator(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)

    plt.savefig("./images/ROC_plot.png")


    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_test,
        './images/result.png')

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    DF = import_data('./data/bank_data.csv')

    perform_eda(DF)

    encoded_DF = encoder_helper(DF,
                                ["Gender",
                                 'Education_Level',
                                 'Marital_Status',
                                 'Income_Category',
                                 'Card_Category'],
                                'Churn')

    x_train, x_test, y_train, y_test = perform_feature_engineering(
        encoded_DF, 'Churn')

    train_models(x_train, x_test, y_train, y_test)
