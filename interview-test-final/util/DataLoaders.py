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
    def load_data(self):
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
        return df

