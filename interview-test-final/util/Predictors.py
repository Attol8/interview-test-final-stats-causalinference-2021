import logging
from abc import ABC, abstractmethod

class Model(ABC):

    def __init__(self):
        super().__init__()
        logging.info('Initializing model')

    # TODO: Feel free to add or change these methods.
    @abstractmethod
    def train(self):
        logging.info('Training model')

    @abstractmethod
    def predict(self):
        logging.info('Doing predictions')


