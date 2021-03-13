# iHerb Content Moderation Model

A model to identify reviews that violate iHerb content rules.

Project Organization
------------

    ├── LICENSE            <- Cookicutter data science project structure license
    ├── README.md          <- The top-level README for developers using this project
    ├── environment.yml    <- Conda environment file with all dependencies 
    ├── data
    │   ├── processed      <- Folder for the final, canonical data sets for modeling
    │   └── raw            <- Folder for the original, immutable data dump
    │
    ├── docs               <- Document describing the project overview, goals and methodology
    │
    ├── models             <- Folder for trained models
    │
    ├── reports            <- Results of modeling 
    │   └── figures        <- Generated graphics and figures to be used in reporting are stored here
    │
    ├── src                <- Source code for use in this project
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── create_data_sets.py <- Script to create all data sets
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── process_all_data.py <- Script to clean and process data
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── build_model.py <- Builds the binary violation/non-violation model 
    │   │   ├── build_multiclass_model.py <- Builds a model that predicts violation categories
    │   │   └── build_multiclass_model_from_binary.py <- Builds a model that takes the binary model output and predicts violation categories 
    │   │    
    │   ├── predictions <- Modeling prediction outputs from each model run as csv and pickle
    │   │       
    │   └── results  <- Script with functions to display metrics and visualization results
    │   │   └── show_model_results.py
        └── run_scripts.py  <- Script for running all data set creation, processing, modeling and results scripts 


## Description
The functions in this project will take raw review data outputs and turn them into processed training and test files which are then input into a modeling script which will generate a trained model and predictions. Finally, there is a script with several functions to show the results of testing the model, including a general accuracy score, confusion matrix, classification report (precision, recall, F1), ROC curve plot and AUC score. 

## Anaconda Environment Installation
An `environment.yml` file is provided in the root directory to allow for easy installation of an environment with all dependencies used in the project. Get the latest version of [Anaconda]('https://docs.anaconda.com/anaconda/install/') to get started. Once Anaconda is installed use the [environment creation command]('https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file') for building environments from .yml files.

```
conda env create -f environment.yml
```

## Running the Model Pipeline

Source File: `src/run_scripts.py`  

All of the major scripts and functions described below are in the `run_scripts.py` file and can be run from there or from the scripts themselves.  

The optimal flow for running the scripts follows below. This will create the data sets and generate both binary and multiclass predictions, combining them in a merged predictions set.
- `data.create_data_sets.create_data()` 
    - Create and process data sets.
- `models.build_model.train_model()` 
    - Trains the binary model and outputs predictions.
- `models.build_multiclass_model_from_binary.train_multiclass_binary_model()` 
    - Takes the predictions data from the binary model, trains a multiclass model on the violations to improve violation category predictions, and outputs the binary and multiclass results together.

## Create Data Sets 

Source File: `src/data/create_data_sets.py`  

Create several data sets that can be used for training and testing the model.

The data creation file is `create_data_sets.py` which contains a `create_data()` function. Running this function will automatically look for 5 raw csv files that are a part of the regular data drops. Modify this section of the function to target different raw source files.

```
def create_data():
    """create balanced, unbalanced, train, validation and test sets
    Input: Path to source csv files for data set
    Output: Saves several csv data sets to /data/processed folder
    """
    start_time = time.time()

    # Import raw data set
    d0 = pd.read_csv("../data/raw/000000000000.csv")
    d1 = pd.read_csv("../data/raw/000000000001.csv")
    d2 = pd.read_csv("../data/raw/000000000002.csv")
    d3 = pd.read_csv("../data/raw/000000000003.csv")
    d4 = pd.read_csv("../data/raw/000000000004.csv")
    df = pd.concat([d0, d1, d2, d3, d4])
    ...
    ...
```

The function will then create several data set variants that can be used for training and testing. They can be modified to produce whatever variants desired. Note that the naming convention is not necessarily indicative of the final intended usage of that data for the model.

- `oot_validation_202008_202009.csv` - An out-of-time validation set from between 2020.08 - 2020.09 This is the time frame after the 2019.01 - 2020.07 time frame for the balanced_test data set and before the start of the promotion, beginning 2020.10.
- `balanced_test.csv` - A data set consisting of a sample of 100k violation and 100k non-violation observations between 2019.01 and 2020.07.
- `unbalanced_train.csv` - A data set consisting of half of the data from 2019.01 - 2020.09.
- `unbalanced_valid.csv` - A data set consisting of the other half of the data from 2019.01 - 2020.09. 

The data sets are then output to the `data/processed/` folder. 

## Data Processing 

Source File: `src/features/process_all_data.py`

The data sets described above are processed before they are finalized, within the `create_data()` function, with the `process_data()` function: 

```
...
...
# Run data cleaning function. This might take a LONG time to run, from .
    df = process_data(df) 
    print('Data cleaning complete.')
...
...
```

`process_data()` is imported from the `process_all_data.py` file and goes through several data processing steps:  
- Cleans and lemmatizes text data
- Removes the observations that were present in the `reviews_violations_english_v5.csv` data
- Pattern matches for UGC name violations, medical words and profanity
- Removes deprecated violations (6, 8, 12)
- Handles emoji violations 
- Ensures all violations are properly identified in the 'ViolationBool' column
- Adds text structural features (string length, word count, average word length)
- Adds date features (year, month, hour)
- Adds language codes as a feature
- NOTE: There are several status printouts for the `clean_data()` script, as it can take quite a while to run

## Build Model 

Source files: `src/models/build_model.py`, `src/models/build_multiclass_model.py`, `src/models/build_multiclass_model_from_binary.py`

Both binary and multiclass models can be built with separate files.

- `build_model.py` - Uses the `train_model()` function to build a binary violation/non-violation classification model
- `build_multiclass_model.py` - Uses the `train_multiclass_model()` function to build a multiclass model that classifies these categories: 
    - 0:'Rewards Code',
    - 1:'Profanity',
    - 2:'Poor Quality/Spam',
    - 3:'Directs Business Away',
    - 4:'Medical Advice',
    - 5:'Shipping and Customs',
    - 6:'Customer Care'

- `build_multiclass_model_from_binary.py` - Functions like the `build_multiclass_model.py`, but takes in the predictions output of the binary model as an input and produces updated multiclass predictions for violations. It will output an `all_predictions.csv` and `all_predictions.pkl` file that has all binary predictions along with the updated multiclass predictions

These functions takes two parameters:  
- `train_path` - Path to the training data
- `test_path` - Path to the test data

### Modeling Steps
These steps are largely the same for both all modeling scripts
- Create test and training data from pandas DataFrames and converts them to the proper TensorFlow format for the model
- Creates the TensorFlow feature columns that will be used in training
- Defines the estimator, a TensorFlow DNNClassifier
- Trains the model on the training set
- Evaluates and returns general accuracy with the test set
- Appends the predictions and associated probabilities to the test set and outputs this to csv and pickle files, which can then be used for further modeling or generating metric scores

## Show Modeling Results 

Source File: `src/results/show_model_results.py`  

The `show_model_results.py` file contains a number of functions for displaying model results. Each of the functions has only one parameter:
- `path` - Path to the `predictions.csv` file generated in the modeling steps above

### Results Functions
Function descriptions:
- `print_confusion_matrix()` - Prints a confusion matrix for the test results
- `show_classification_report()` - Shows a classification report for the precision, recall and F1 scores for each of the classes for the binary model
- `show_multiclassification_report()` - Shows a classification report for the precision, recall and F1 scores for each of the classes for the multi-class model
- `plot_roc_curve()` - Shows the ROC curve of the binary test results
- `print_auc_score()` - Shows the AUC score of the binary test results



## Cookiecutter Project Format License
The MIT License (MIT)
Copyright (c) 2021, The Spur Group

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
