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
                                  'format_data/26.pretrain_data_processing/01.insitu_and_basin_average/10.HydroBasin/GLOBAL_level_8_global.nc'),
    'train_date_list': ["1999-01-01", "2016-12-31"],
    'val_date_list': ["1998-01-01", "1998-12-31"],
    'test_date_list': ["1980-01-01", "1997-12-31"],

    'time_series_variables': ["P", "SWd", "RelHum", "Tmax", "Tmin", ],  # "LWd", "PET","SpecHum", "Temp", "Wind"
    'target_variables': [],
    'static_variables': ['meanP', 'seasonality_P', 'seasonality_PET', 'meanTa', 'snow_fraction', 'snowfall_fraction',
                         'meanelevation', 'meanslope', 'aspectcosine', 'HWSD_clay', 'HWSD_sand', 'HWSD_silt',
                         'porosity', 'Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'soil_depth',
                         'NDVI', 'grassland_frac', 'forest_frac', 'soil_erosion', 'catchsize'],  # 'HWSD_gravel',
    'static_variables_category': [],  # 'landcover', 'lithology'
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        'climate': ['meanP', 'seasonality_P', 'seasonality_PET', 'meanTa', 'snow_fraction', 'snowfall_fraction'],
        'topography': ['meanelevation', 'meanslope', 'aspectcosine', 'soil_depth'],  #
        'soil': ['HWSD_clay', 'HWSD_sand', 'HWSD_silt'],  # 'HWSD_gravel',
        'geology': ['Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'porosity'],
        'vegetation': ['NDVI', 'grassland_frac', 'forest_frac'],
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize'],
    'negative_value_variables': ["Tmax", "Tmin", "Temp", "meanTa", "permeability", "permeability_permafrost", "Permeability_no_permafrost", "aspectcosine", "meanelevation"],
}

tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids.values[:100]

config_dataset_global_hydrobasin_level08 = SimpleNamespace(**tmp)
