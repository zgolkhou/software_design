Last login: Fri Mar 12 12:06:11 on console
(base) Zachs-iMac:~ zach$ vi /Users/zach/Desktop/content-moderation-model/README.md 

# Content Moderation Model

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

