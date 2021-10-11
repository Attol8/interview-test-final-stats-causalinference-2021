import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
import numpy as np
import yaml
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

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
            except yaml.YAMLError as exc:
                print(exc)

        
        #balance class
           
        over_SMOTE = SMOTE(random_state = 11)
        over_SMOTETomek = SMOTETomek(random_state=11)
        under = RandomUnderSampler(sampling_strategy=0.5)

        data_pipeline = ColumnTransformer([
            ('numerical', StandardScaler(), numerical_features),
            ('categorical', OneHotEncoder(handle_unknown='ignore'), categorical_features),
            #('ordinal', OrdinalEncoder(handle_unknown='ignore'), ordinal_features)
        ])

        pipeline = Pipeline([
            ('data_pipeline', data_pipeline),
            ('over', over_SMOTETomek),
            #('under', under), #Step2 - clean and transform data
            ('clf', self.model) #step3 - classifier
        ])
        return pipeline

    def get_param_grid(self):
        if self.model_name == 'LogReg':
            param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
        
        if self.model_name == 'RF':
            param_grid = { 
            'clf__n_estimators': [200, 500],
            'clf__max_depth' : np.linspace(3, 35, 33),
            }
        
        return param_grid

def report_to_df(report):
    split_string = [x.split(' ') for x in report.split('\n')]
    column_names = ['']+[x for x in split_string[0] if x!='']
    values = []
    for table_row in split_string[1:-1]:
        table_row = [value for value in table_row if value!='']
        if table_row!=[]:
            values.append(table_row)
    for i in values:
        for j in range(len(i)):
            if i[1] == 'avg':
                i[0:2] = [' '.join(i[0:2])]
            if len(i) == 3:
                i.insert(1,np.nan)
                i.insert(2, np.nan)
            else:
                pass
    report_to_df = pd.DataFrame(data=values, columns=column_names)
    return report_to_df


        





