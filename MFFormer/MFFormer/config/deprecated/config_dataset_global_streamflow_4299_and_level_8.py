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
    'input_nc_file': os.path.join(server_dir,
                                  'format_data/26.pretrain_data_processing/01.insitu_and_basin_average/10.HydroBasin/GLOBAL_4299_and_level_8.nc'),
    'train_date_list': ["1999-01-01", "2016-12-31"],
    'val_date_list': ["1998-01-01", "1998-12-31"],
    'test_date_list': ["1980-01-01", "1997-12-31"],

    'time_series_variables': ['P', 'PET', 'Tmax', 'Tmin', 'Runoff'],  # 'Wind',
    'target_variables': [],
    'static_variables': ['meanP', 'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction', 'meanTa',
                         'meanelevation', 'meanslope', 'aspectcosine', 'HWSD_clay', 'HWSD_sand', 'HWSD_silt',
                         'HWSD_gravel', 'porosity', 'permeability', 'carbonate_sedimentary_rocks_frac', 'soil_depth',
                         'NDVI', 'grassland_frac', 'forest_frac', 'soil_erosion', 'catchsize'],  #
    'static_variables_category': [],  # 'landcover', 'lithology'
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        # 'landform':['glaciers', 'permafrost'],
        'topography': ['meanelevation', 'meanslope', 'aspectcosine'],  #
        'soil': ['HWSD_clay', 'HWSD_sand', 'HWSD_silt', 'HWSD_gravel'],  #
        'geology': ['permeability', 'carbonate_sedimentary_rocks_frac', 'soil_depth', 'porosity'],
        'vegetation': ['NDVI', 'grassland_frac', 'forest_frac'],
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize'],
    'negative_value_variables': ["Tmax", "Tmin", "Temp", "meanTa", "permeability", "permeability_permafrost",
                                 "Permeability_no_permafrost", "aspectcosine", "meanelevation"],

}

tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids.values

# # load the json index file
# import json
# import numpy as np
# json_file = "/data/jql6620/local/30.MFFormer/MFFormer/output/pretrain_MFFormer_GLOBAL_4299_cv_kfold0/index/index_cv_num_kfold_5_nth_kfold_0_test_GLOBAL_4299.json"
# with open(json_file, 'r') as f:
#     test_index = np.array(json.load(f))
# test_index = test_index[test_index < 10]
# tmp['station_ids'] = tmp['station_ids'][test_index]

config_dataset_global_4299_and_level_8 = SimpleNamespace(**tmp)
