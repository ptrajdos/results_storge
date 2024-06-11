import unittest
import numpy as np
from results_storage.results_storage import ResultsStorage
import tempfile
import pickle

class ResultsStorageTest(unittest.TestCase):

    def test_init_dims_coords(self):
        a = ResultsStorage.init_dims_coords(dims=("A","B"), coords={"A":[1,2,3], "B":[4,5,6]},fill_value=0, name="a")
        self.assertIsNotNone(a, "Initialised object is None")
        self.assertTrue( a.shape == (3,3), "Wrong dimension")
        self.assertTrue( np.allclose( a , np.zeros((3,3))), "Wrong fill")

        m0 = np.mean(a,axis=0)
        m1 = np.mean(a,axis=1)

        self.assertFalse( np.any( np.isnan(m0)) , "Nans in mean over axis 0" )
        self.assertFalse( np.any( np.isnan(m1)) , "Nans in mean over axis 1" )

    def test_pickle(self):
        X  = np.random.random((10,5))
        a = ResultsStorage(X, name="A")

        tmp_filename  = tempfile.mkstemp()[1]
        
        tmp_file_handler_w = open(tmp_filename, "wb")

        pickle.dump(a, tmp_file_handler_w)
        tmp_file_handler_w.close()

        tmp_file_handler_r = open(tmp_filename, "rb")
        ac = pickle.load(tmp_file_handler_r)
        tmp_file_handler_r.close()

        self.assertTrue( np.allclose( a, ac),"Different values get from pickle")




        

if __name__ == '__main__':
    unittest.main()