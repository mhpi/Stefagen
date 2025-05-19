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

basin_file = os.path.join(os.path.split(MFFormer.__file__)[0], "datasets/GLOBAL_streamflow/GLOBAL_streamflow_selection.txt")
station_ids = pd.read_csv(basin_file, header=None, dtype=str).values[:, 0].tolist()

tmp = {
    'input_nc_file': os.path.join(server_dir, 'format_data/26.pretrain_data_processing/01.insitu_and_basin_average/01.Hylke_global_streamflow/GLOBAL_4299.nc'),
    'train_date_list': ["1999-01-01", "2016-12-31"],
    'val_date_list': ["1998-01-01", "1998-12-31"],
    'test_date_list': ["1980-01-01", "1997-12-31"],

    'time_series_variables': ['P', 'Tmax', 'Tmin', 'PET',  'Runoff'], #'Wind',
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

config_dataset_global_4299_selection = SimpleNamespace(**tmp)
