
from typing import Any

import xarray as xr
from xarray.core import dtypes
import numpy as np

class ResultsStorage(xr.DataArray):
    def __init__(self, data: Any = dtypes.NA, coords= None, dims= None, name = None,
                  attrs = None, indexes = None, fastpath: bool = False) -> None:
        super().__init__(data, coords, dims, name, attrs, indexes, fastpath)

    __slots__ = ()#Subclass of DataAray do not define new attributes

    @classmethod
    def init_dims_coords(cls,dims, coords:dict,fill_value=np.nan, **kwargs):
        """
        Initialises the result storage using dims and coords. 
        No array or dataframe is needed as a source.
        Fils structure with given value
        Arguments:
        ----------
        dims -- iterable with dimension names
        coords: dict -- coordinates dict
        fill_values -- value the new structire to be filled. Default numpy.nan
        **kwargs -- additional arguments to xarray.DataArray

        Returns:
        --------
        New object of ResulstStorage
        """
        dim_cords_lengths = tuple( len(coords[dim_name]) for dim_name in dims)
        empty_array  = np.zeros((dim_cords_lengths))
        empty_array[:] = fill_value
        obj = cls(data=empty_array, dims = dims, coords=coords,**kwargs )
        return obj
    
    @classmethod
    def init_coords(cls,coords:dict,fill_value=np.nan, **kwargs):
        """
        Initialises the result storage using coords. 
        No array or dataframe is needed as a source.
        Fils structure with given value
        Arguments:
        -----------
        coords -- coords to initialise the objece
        fill_values -- value the new structire to be filled. Default numpy.nan
        **kwargs -- additional arguments to xarray.DataArray

        Returns:
        --------
        New object of ResulstStorage
        """

        dims = [ k for k in coords ] #dimension names extracted
        return cls.init_dims_coords(dims,coords,fill_value,**kwargs)
        
    
    @staticmethod
    def coords_need_recalc(obj:xr.DataArray, dim_name):
        """
        Check which coordinates inside one dimension contains at least one nan.
        If so then return this coord
        Arguments:
        ---------
        obj -- DataArray object
        dim_name -- dimension name

        Yields: 
        -------
        coordinates that needs to be recalced
        """
        for a in obj[dim_name].values:
            if not bool(np.any(np.isnan( obj.loc[{dim_name:a}]))):
                continue
            yield a
    