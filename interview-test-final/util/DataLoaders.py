import logging
from abc import ABC, abstractmethod
import os.path
import pandas as pd

class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self, filename):
        logging.info('Checking file exists.')

        if not os.path.isfile(filename):
            logging.error('File does not exist')
            # TODO: raise exception (comment: I raise the excpetion in the implemented class)
        else:
            logging.info('Found file: ' + filename)
            
class FileDataLoader(AbstractDataLoader):

    # Initialization
    def __init__(self, filename: str):
        super().__init__()
        logging.info('Initializing Data Loading')
        self.filename = filename

    # Load data from file and return data
    def load_data(self, impute_nas = None, upsample_df = None):
        # TODO: Check file exists
        try: 
            with open(self.filename) as f:
                logging.info('Found file: ' + self.filename)
        except FileNotFoundError:
            raise

        # TODO: Load data from file
        logging.info('Loading data using pandas')
        df = pd.read_csv(self.filename)

        # TODO: Return your data object here

        if impute_nas is not None:
            features_to_impute = ['salary', 'mins_beerdrinking_year', 'mins_exercising_year', 'tea_per_year', 'coffee_per_year']
            for feature in features_to_impute:
                df[feature] = df[feature].interpolate(method='polynomial', order=2)
                
        if upsample_df is not None:
            pass

        return df

