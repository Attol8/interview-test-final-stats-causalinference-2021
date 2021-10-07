import unittest
import sys
sys.path.append('interview-test-final')
from util.DataLoaders import FileDataLoader
class Test(unittest.TestCase):

    def test_load_data_invalid_file(self):
        invalid_filename = '..\data\dataset_experimentations.csv'
        data_loader = FileDataLoader(invalid_filename)
        self.assertRaises(FileNotFoundError, data_loader.load_data)

if __name__ == "__main__":
    unittest.main()