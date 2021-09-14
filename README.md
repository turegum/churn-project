# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
### The purposes of the project where:
* Refactor **churn_notebook.ipynb** code to follow best practices
* Write unit tests for every function

### Directory layout
    .
    ├── data                               # Bank data file in .csv format
    ├── images                             # EDA and result plots separated
    │   ├── eda                            # Exploratory plots
    │   ├── results                        # Result plots
    ├── logs                               # Logs file is created here
    ├── models                             # RF and LR models are saved here
    ├── LICENSE
    ├── README.md
    ├── churn_library.py                   # Refactored code
    ├── churn_script_logging_and_tests.py  # Unit tests for every function     
    ├── constants.py                       # Constants are separated from scripts            
    └── requirements.txt                   

## Running Files
The way you perform unit testing:
```python
python churn_script_logging_and_tests.py
```
You can find the log file in the logs subdirectory after script execution.
Also you're welcome to change test parameters in the end of the script.

You can run the main script this way:
```python
python churn_library.py
```
You can find all the results in the corresponding directories mentioned above.

## Dependencies
Here is a list of needed libraries:
'''
joblib==1.0.1
shap==0.39.0
pandas==1.1.4
seaborn==0.11.2
matplotlib==3.4.3
numpy==1.20.3
scikit_learn==0.24.2
'''

## Lincense
[GNU 3.0] (https://www.gnu.org/licenses/gpl-3.0.en.html)



