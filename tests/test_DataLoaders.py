import unittest
import pandas as pd
import sys
sys.path.append('interview-test-final')
from util.DataLoaders import FileDataLoader

class Test(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv('data\dataset_experimentation.csv')

    def test_load_data_invalid_file(self):
        #wrong name because 's' at the end of 'experimentation'
        invalid_filename = '..\data\dataset_experimentations.csv' 
        data_loader = FileDataLoader(invalid_filename)
        self.assertRaises(FileNotFoundError, data_loader.load_data)

    def test_column_names(self):
        required_names = ['user_id','age','workclass','salary','education_rank','marital-status','occupation','race','sex','mins_beerdrinking_year','mins_exercising_year','works_hours','tea_per_year','coffee_per_year','great_customer_class','test','spent_17','spent_18']
        self.assertListEqual(list(self.df.columns), required_names)

if __name__ == "__main__":
    unittest.main()