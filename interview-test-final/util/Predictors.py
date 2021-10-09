import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import numpy as np
import yaml
class AbstractModel(ABC):

    def __init__(self):
        super().__init__()
        logging.info('Initializing model')

    # TODO: Feel free to add or change these methods.
#    @abstractmethod
#    def train(self):
#        logging.info('Training model')

#   @abstractmethod
#    def predict(self):
#        logging.info('Doing predictions')

class Model(AbstractModel):
    """

    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.available_models =  {'LogReg' : LogisticRegression(max_iter = 1000), 'RF' : RandomForestClassifier()}
        self.load_model()

    def load_model(self): 
        try:
            self.model = self.available_models[self.model_name]
        except: 
            print('Selected model is not available')

    def get_pipeline(self):

        #TODO: load from config file features categories
        with open("util/config.yaml", "r") as config:
            try:
                config_dict = yaml.safe_load(config)

                categorical_features = config_dict['categorical_features']
                numerical_features = config_dict['numerical_features']
                ordinal_features = config_dict['ordinal_features']
            except yaml.YAMLError as exc:
                print(exc)

        #TODO: add numerical transformers
        data_pipeline = ColumnTransformer([
            ('numerical', StandardScaler(), numerical_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            #('ordinal', OrdinalEncoder(handle_unknown='ignore'), ordinal_features)
        ])

        pipeline = Pipeline([
            ('data_pipeline', data_pipeline), #Step1 - clean and transform data
            ('clf', self.model) #step2 - classifier
        ])
        return pipeline

    def get_param_grid(self):
        if self.model_name == 'LogReg':
            param_grid = {
                'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            }
        
        if self.model_name == 'RF':
            param_grid = { 
            'clf__n_estimators': [200, 500],
            'clf__max_depth' : np.linspace(3, 35, 33),
            }
        
        return param_grid




        





