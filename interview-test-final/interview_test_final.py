import logging
import json
from collections import namedtuple
from util.DataLoaders import FileDataLoader
from util.Predictors import Model, report_to_df
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import os
from sklearn.model_selection import GridSearchCV # For optimization
import yaml
import pandas as pd
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as imblearnPipeline
from joblib import dump, load

def sort_file_paths(project_name: str):
    # figure out the path of the file we're runnning
    runpath = os.path.realpath(__file__)
    # trim off the bits we know about (i.e. from the root dir of this project)
    rundir = runpath[:runpath.find(project_name) + len(project_name) + 1]
    # change directory so we can use relative filepaths
    os.chdir(rundir + 'interview-test-final')

def load_config():
    run_configuration_file = '../resources/interview-test-final.json'
    with open(run_configuration_file) as json_file:
        json_string = json_file.read()
        run_configuration = json.loads(json_string,
                                       object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    return run_configuration

if __name__ == '__main__':
    # Initialize logging
    logging.basicConfig(format="%(asctime)s;%(levelname)s;%(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    logging.info('Starting classification program')

    # Actions: get into working directory, load project config, create dated directories
    sort_file_paths(project_name='interview-test-final-stats-causalinference-2021')
    run_configuration = load_config()


    # TODO: Load the data by instantiating the FileDataLoader, handle file doesn't exist.
    data_loader = FileDataLoader('../data/dataset_experimentation.csv')  # Candidate , instantiate your class here
    df = data_loader.load_data(impute_nas = True) #

    # TODO: Do the rest of your work here, or in other classes that are called here.

    #name of the experiment run for comparison later
    experiment_name = 'baseline'

    #load target variable and variables to drop
    with open("util/config.yaml", "r") as config:
        try:
            config_dict = yaml.safe_load(config)

            drop_features = config_dict['drop_features']
            target_feature = config_dict['target_feature']
        except yaml.YAMLError as exc:
            print(exc)

    #drop na values 
    df = df.dropna()

    #split df into train and test sets
    X = df.drop(drop_features + target_feature, axis=1)
    y = df[target_feature[0]]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=.3, random_state=42)

    #names of models to run
    model_names = ['RF', 'LogReg']

    results, model_names_results = [], []
    for model_name in model_names:         
        model = Model(model_name).model # Candidate, instantiate your class here
        
        #create pipeline to Sequentially apply a list of transforms and a final estimator (model)
        pipe = Model(model_name).get_pipeline()
        
        #uncomment to perform grid search to find optimal parameters
        #logging.info(f'Tuning {model_name} hyper-parameters')
        #parameters = Model(model_name).get_param_grid()
        #grid = GridSearchCV(pipe, parameters, cv=3, n_jobs = -1, scoring = 'accuracy', verbose = 2).fit(X_train, y_train)

        #check model performance with cross validation
        result = cross_val_score(pipe, X_train, y_train,  scoring = 'roc_auc', cv=3)

        #results of the best estimator - check model performance with cross validation
        logging.info(f'Cross validating {model_name} model')
        print('')
        print(f'{model_name} cross val roc_auc score: {round(result.mean(), 2)}\n')

        #fit the model to the entire training data
        pipe.fit(X_train, y_train)

        #predict on test set
        logging.info(f'Training {model_name} model')

        #save classifications reports
        logging.info(f'Saving {model_name}')

        dump(pipe, f'../interview-test-final/util/trained_models/{model_name}_{experiment_name}.joblib')



    logging.info('Completed program')
