import unittest

import pandas as pd

# Alexander imports
from sys import path
from os.path import dirname as dir
path.append(dir(path[0]))

import preprocessing
 
class TestPreprocessing(unittest.TestCase):
 
    def setUp(self):
        pass

    def test_label_encoder(self):
        """Test that LabelEncoder works as intended for a normal case."""
        encoder = preprocessing.LabelEncoder()
        data = [{'Sex': 'Male'},{'Sex': 'Female'}]
        df = pd.DataFrame(data)
        new_df = encoder.fit_transform(df)
        expected_data = [{'Sex': 1},{'Sex': 0}]
        self.assertEqual(new_df.to_dict('records'), expected_data)
 
    def test_label_encoder_encodings(self):
        """Test that attribute 'encodings' of LabelEncoder is correct."""
        encoder = preprocessing.LabelEncoder()
        data = [{'Sex': 'Male'},{'Sex': 'Female'}]
        df = pd.DataFrame(data)
        new_df = encoder.fit(df)
        sorted_a = sorted(encoder.encodings['Sex'])
        sorted_b = sorted(['Male', 'Female'])
        self.assertEqual(sorted_a, sorted_b)
 
if __name__ == '__main__':
    unittest.main()