
from typing import Any

import xarray as xr
from xarray.core import dtypes
import numpy as np

class ResultsStorage(xr.DataArray):
    def __init__(self, data: Any = dtypes.NA, coords= None, dims= None, name = None,
                  attrs = None, indexes = None, fastpath: bool = False) -> None:
        super().__init__(data, coords, dims, name, attrs, indexes, fastpath)

    @classmethod
    def init_dims_coords(cls,dims, coords,fill_value=np.nan, **kwargs):
        dim_cords_lengths = tuple( len(coords[dim_name]) for dim_name in dims)
        empty_array  = np.zeros((dim_cords_lengths))
        empty_array[:] = fill_value
        obj = cls(data=empty_array, dims = dims, coords=coords,**kwargs )
        return obj
    