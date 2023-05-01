import azimuth
import azimuth.model_comparison
import numpy as np
import unittest
import pandas
import os
dirname, filename = os.path.split(os.path.abspath(__file__))

class SavedModelTests(unittest.TestCase):
    """
    This unit test checks that the predictions for 1000 guides match the predictions we expected in Nov 2016.
    This unit test can fail due to randomness in the model (e.g. random seed, feature reordering).
    """

    def test_predictions_nopos(self):
        df = pandas.read_csv(os.path.join(dirname, '1000guides.csv'), index_col=0)
        predictions = azimuth.model_comparison.predict(np.array(df['guide'].values), None, None)
        if not np.allclose(predictions, df['truth nopos'].values, atol=1e-3):
            failure_count = 0
            for i, (pred, truth) in enumerate(zip(predictions, df['truth nopos'].values)):
                if not np.isclose(pred, truth, atol=1e-3):
                    failure_count += 1
                    print(f"Row {i}: nopos prediction={pred}, truth nopos={truth}")
            print(f"Total failure count: {failure_count} out of {len(df['truth nopos'])}")
            self.fail("The predictions and truth nopos values are not close enough")


    def test_predictions_pos(self):
        print('\n')
        df = pandas.read_csv(os.path.join(dirname, '1000guides.csv'), index_col=0)
        predictions = azimuth.model_comparison.predict(np.array(df['guide'].values), np.array(df['AA cut'].values), np.array(df['Percent peptide'].values))
        if not np.allclose(predictions, df['truth pos'].values, atol=1e-3):
            failure_count = 0
            for i, (pred, truth) in enumerate(zip(predictions, df['truth pos'].values)):
                if not np.isclose(pred, truth, atol=1e-3):
                    failure_count += 1
                    print(f"Row {i}: pos prediction={pred}, truth pos={truth}")
            print(f"Total failure count: {failure_count} out of {len(df['truth pos'])}")
            self.fail("The predictions and truth pos values are not close enough")


        
if __name__ == '__main__':
    unittest.main()
