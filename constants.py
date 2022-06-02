data_path = './data/bank_data.csv'

eda_images_dir = './images/eda/'
eda_image_dict = {
    "churn_png" :'churn.png',
    "customer_age" : 'customer_age.png',
    "marital_status" : 'marital_status.png',
    "trans_distro" : 'transaction_distribution.png',
    "corr_heatmap" : 'correlation_heatmap.png'
    }
for key, value in eda_image_dict.items():
    eda_image_dict[key] = eda_images_dir + value
required_eda_ouputs = set([value for value in eda_image_dict.values()])


results_image_dir = './images/results/'
results_image_dict = {
    'lr' : 'Logistic Regression.png',
    'rf' : 'Random_forest.png',
    'roc' : 'ROC_curve.png',
    'feature_importance' : 'feature_importance_plot.png'
}
for key, value in results_image_dict.items():
    results_image_dict[key] = results_image_dir + value
required_images = [value for value in results_image_dict.values()]


models_dir = './models/'
models_dict = {
    'lr' : 'logistic_model.pkl',
    'rf' : 'rfc_model.pkl'
}
for key, value in models_dict.items():
    models_dict[key] = models_dir + value
required_models = [value for value in models_dict.values()]

required_training_outputs = set(required_models + required_images)

non_cat_keep_cols =  ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                ]

cat_columns = [
            'Gender',
            'Education_Level',
            'Marital_Status',
            'Income_Category',
            'Card_Category'                
        ]