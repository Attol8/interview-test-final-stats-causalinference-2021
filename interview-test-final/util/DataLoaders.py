import logging
from abc import ABC, abstractmethod
import os.path

class AbstractDataLoader(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def load_data(self, filename):
        logging.info('Checking file exists.')

        if not os.path.isfile(filename):
            logging.error('File does not exist')
            # TODO: raise exception
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

        # TODO: Load data from file
        logging.info('Loading data using pandas')

        # TODO: Return your data object here
        return 0

