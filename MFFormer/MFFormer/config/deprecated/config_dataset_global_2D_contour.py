"""
errors or too much NaN: 'aridity', 'permafrost',
remove: 'glaciers',
"""
import os
import MFFormer
import pandas as pd
from types import SimpleNamespace
import xarray as xr
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

tmp = {
    'input_nc_file': '/data/jql6620/local/30.MFFormer/examples/07.2D_Contour/output/2D_contour.nc',
    'train_date_list': ["1999-01-01", "2016-12-31"],
    'val_date_list': ["1998-01-01", "1998-12-31"],
    'test_date_list': ["1980-01-01", "1997-12-31"],

    'time_series_variables': ['P', 'Tmax', 'Tmin', 'PET',  'Runoff'], #'Wind',
    'target_variables': [],
    'static_variables': ['meanP', 'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction', 'meanTa',
                         'NDVI', 'meanelevation', 'meanslope', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt',
                         'soil_depth', 'permeability', 'porosity', 'carbonate_sedimentary_rocks_frac', 'catchsize',
                         'grassland_frac', 'forest_frac', 'soil_erosion', 'aspectcosine'],  #
    'static_variables_category': [],  # 'landcover', 'lithology'
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        # 'landform':['glaciers', 'permafrost'],
        'topography': ['meanelevation', 'meanslope', 'aspectcosine'],
        'soil': ['HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt', 'soil_depth'],
        'geology': ['permeability', 'carbonate_sedimentary_rocks_frac', 'porosity'],
        'vegetation': ['NDVI', 'grassland_frac', 'forest_frac'],
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize'],
    'negative_value_variables': ["Tmax", "Tmin", "Temp", "meanTa", "permeability", "permeability_permafrost",
                                 "Permeability_no_permafrost", "aspectcosine", "meanelevation"],

}

tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids.values

config_dataset_2D_contour = SimpleNamespace(**tmp)
