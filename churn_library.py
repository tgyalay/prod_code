''''
This library contains two independent functions and a class for running
thorugh the procedure of analyzing the data. 
'''

# import libraries
import os
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve, classification_report
import shap

import constants


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    return pd.read_csv(pth)


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
    final_df = df.copy()
    for category in category_lst:
        category_grouped = df.groupby(category).mean()['Churn'].rename(
            '_'.join([category,response]))
        final_df = final_df.join(category_grouped, on=category)

    return final_df


class ChurnAnaysis():
    '''
    used to perform analysis for customer churn. 

    input:
            df: pandas dataframe
    
    attributes:
            df pandas dataframe
    '''
    def __init__(self, pth):
        self.df = import_data(pth)
        self.df['Churn'] = self.df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

    def perform_eda(self):
        '''
        perform eda on df and save figures to images folder
        input:
                None

        output:
                None
        '''

        image_dict = constants.eda_image_dict

        plt.figure(figsize=(20,10)) 
        self.df['Churn'].hist()
        plt.savefig(image_dict['churn_png'])
        plt.close()

        plt.figure(figsize=(20,10)) 
        self.df['Customer_Age'].hist()
        plt.savefig(image_dict['customer_age'])
        plt.close()

        plt.figure(figsize=(20,10)) 
        self.df.Marital_Status.value_counts('normalize').plot(kind='bar')
        plt.savefig(image_dict['marital_status'])
        plt.close()

        plt.figure(figsize=(20,10))
        sns.histplot(self.df['Total_Trans_Ct'], stat='density', kde=True)
        plt.savefig(image_dict['trans_distro'])
        plt.close()

        plt.figure(figsize=(20,10)) 
        sns.heatmap(self.df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
        plt.gcf().tight_layout()
        plt.savefig(image_dict['corr_heatmap'])
        plt.close()

    def perform_feature_engineering(self, response='Churn'):
        '''
        input:
                response: string of response name [optional argument that could be used for naming variables or index y column]

        attributes:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        '''

        cat_columns = constants.cat_columns

        keep_cols = (constants.non_cat_keep_cols 
                     + [category+'_'+response 
                        for category in  constants.cat_columns
                        ]
                    )

        self.df = encoder_helper(self.df,
                                 category_lst=cat_columns,
                                 response=response)

        y = self.df['Churn']
        X = self.df[keep_cols]

        self.X_train, self.X_test, self.y_train, self.y_test =  (
            train_test_split(X, y, test_size= 0.3, random_state=42)
            )

    def classification_report_image(self,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''

        plt.rc('figure', figsize=(5, 5))
        #plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
        plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.gcf().tight_layout()
        plt.savefig(constants.results_image_dict['rf'])
        plt.close()

        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
        plt.axis('off')
        plt.gcf().tight_layout()
        plt.savefig(constants.results_image_dict['lr'])
        plt.close()

    def feature_importance_plot(self, model, X_data, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = model.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [X_data.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20,5))

        # Create plot title
        plt.title("Feature Importance")
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(X_data.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(X_data.shape[1]), names, rotation=90)
        plt.gcf().tight_layout()
        plt.savefig(output_pth)

    def train_models(self):
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
        # Use a different solver if the default 'lbfgs' fails to converge
        # Reference: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

        param_grid = { 
            'n_estimators': [200, 500],
            'max_features': ['auto', 'sqrt'],
            'max_depth' : [4,5,100],
            'criterion' :['gini', 'entropy']
        }

        cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)

        lrc.fit(self.X_train, self.y_train)

        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)

        y_train_preds_lr = lrc.predict(self.X_train)
        y_test_preds_lr = lrc.predict(self.X_test)

        self.classification_report_image(
            y_train_preds_lr=y_train_preds_lr,
            y_train_preds_rf=y_train_preds_rf,
            y_test_preds_lr=y_test_preds_lr,
            y_test_preds_rf=y_test_preds_rf
            )
                    
        lrc_plot = plot_roc_curve(lrc, self.X_test, self.y_test)
        
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        plot_roc_curve(cv_rfc.best_estimator_,
                       self.X_test, self.y_test,
                       ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(constants.results_image_dict['roc'])

        joblib.dump(cv_rfc.best_estimator_, constants.models_dict['rf'])
        joblib.dump(lrc, constants.models_dict['lr'])

        self.feature_importance_plot(
            cv_rfc.best_estimator_, 
            self.X_train,
            constants.results_image_dict['feature_importance'])