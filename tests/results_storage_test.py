import unittest
import numpy as np
from results_storage.results_storage import ResultsStorage
import tempfile
import pickle
import xarray as xr

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

    def test_init_coords(self):

        coords={"A":[1,2,3], "B":[4,5,6]}

        a = ResultsStorage.init_coords(coords,fill_value=0, name="a")
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

    def test_coords_need_recalc(self):
        coords1 = {
        "A":["a{}".format(i) for i in range(2)],
        "B":["b{}".format(i) for i in [4,5,6]],
        "C":["b{}".format(i) for i in [7,8,9]],
        }

        coords2 = {
        "A":["a{}".format(i) for i in range(3)],
        "B":["b{}".format(i) for i in [4,5,6]],
        "C":["b{}".format(i) for i in [7,8,9]],
        }
        a = ResultsStorage.init_dims_coords(dims=("A","B","C"), coords=coords1,fill_value=0, name="a")
        a[:] = 1   
        b = ResultsStorage.init_dims_coords(dims=("A","B","C"), coords=coords2,fill_value=1, name="b")

        c = xr.merge([a,b])["a"]

        cnt = 0

        for a in ResultsStorage.coords_need_recalc(c,"A"):
            cnt+=1

        self.assertTrue(cnt == 1, "Wrong number of recalcs" ) 

    def test_merge(self):
        coords1 = {
        "A":["a{}".format(i) for i in range(2)],
        "B":["b{}".format(i) for i in [4,5,6]],
        "C":["b{}".format(i) for i in [7,8,9]],
        }

        coords2 = {
        "A":["a{}".format(i) for i in range(3)],
        "B":["b{}".format(i) for i in [4,5,6]],
        "C":["b{}".format(i) for i in [7,8,9]],
        }
        a = ResultsStorage.init_dims_coords(dims=("A","B","C"), coords=coords1,fill_value=0, name="a")
        a[:] = np.random.random( (2,3,3) )   
        b = ResultsStorage.init_dims_coords(dims=("A","B","C"), coords=coords2,fill_value=1, name="b")

        merged = ResultsStorage.merge_with_loaded(a,b)
        self.assertIsNotNone(merged, "Merged object is none")
        self.assertIsInstance(merged, xr.DataArray, "Not an instance of DataArray")
        self.assertTupleEqual( merged.shape, (3,3,3), "Wrong shape of merged object")

        for a_name in a["A"].values:
            for b_name in a["B"].values:
                for c_name in a["C"].values:
                    self.assertTrue(np.allclose(a.loc[{"A":a_name, "B":b_name, "C":c_name}],
                                                 merged.loc[{"A":a_name, "B":b_name, "C":c_name}] ), "Chunks not the same!")

        
        self.assertTrue( np.all( np.isnan(merged.loc[{"A":'a2'}]) ), "Only nans should be here" )


        

if __name__ == '__main__':
    unittest.main()