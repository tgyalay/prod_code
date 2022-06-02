# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims to predict whether a customer is going to churn, by performing exploratory data analysis (EDA), feature engineering, and a model search with evaluation at the end.

## Files and data description
Overview of the files and data present in the root directory. 
.
├── Guide.ipynb     # Given: getting started and t/s tips
├── churn_notebook.ipynb # Given: Original, unclean code
├── churn_library.py     # Library of functions/class used
├── churn_script_logging_and_test.py # Main script
├── constants.py         # File paths and column name storage
├── README.md            # ToDo: Provides project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models

## Running Files
### Testing the files
The individual steps of the script can be tested using `pytest -p no:logging churn_script_logging_and_test.py` or by using the test feature in VS Code. Logging for these tests are output to `logs/churn_library.log` when the `-p no:logging` flag is added. 
### Conducting the Analysis
The main script, `churn_script_logging_and_test.py` can either be imported and the `main()` function run, or the script can be run directly and `main()` will be called. 