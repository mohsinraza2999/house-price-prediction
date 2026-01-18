import unittest
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_pipeline import data_preparation
import pandas as pd

class DataProcessing(unittest.TestCase):
    def setUp(self)-> None:
        self.path="data/raw/"
        self.name="housing.csv"
        self.data=[' 0.02729   0.00   7.070  0  0.4690  7.1850  61.10  4.9671   2  242.0  17.80 392.83   4.03  34.70']

    def test_read(self)->None:

        read_object=data_preparation.Readcsv(self.path)
        data=read_object.read(self.name)
        self.assertIsNotNone(data,"Data is None")
        self.assertEqual(data.__class__,pd.DataFrame,"The Data is not Pandas Series")

    def test_clean(self)->None:

        clean_object=data_preparation.CleanData(self.path)
        data=clean_object.seprate_columns(data=self.data)

        assert data != None
        if isinstance(data, list):
            self.assertEqual(len(data[0]), 14)  # 14 values in your string
        else:
            self.fail("Unexpected return type from seprate_columns")

        
        


if __name__=="__main__":
    unittest.main()

