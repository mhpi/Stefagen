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
                                  'format_data/30.MFFormer/GLOBAL_3434_3248_additional_1952.nc'),
    'train_date_list': ["1999-01-01", "2016-12-31"],
    'val_date_list': ["1998-01-01", "1998-12-31"],
    'test_date_list': ["1980-01-01", "1997-12-31"],

    'time_series_variables': ["P", "SWd", "RelHum", "Tmax", "Tmin", ],  # "LWd", "PET","SpecHum", "Temp", "Wind"
    'target_variables': [],
    'static_variables': ['MSWEP_P', 'MSWX_PET', 'MSWX_Tmean', 'MSWX_SWd',
                         'GMTED_elevation', 'GMTED_slope', 'GMTED_aspectcosine', 'HWSD_clay', 'HWSD_sand', 'HWSD_silt',
                         'porosity', 'Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'soil_depth',
                         'MODIS_NDVI', 'grassland_fraction_2018', 'forest_fraction_1992', 'forest_fraction_2018',
                         'catchsize', 'population_density_2000_buffer', 'gdp_2000_buffer', 'intact_forest'], #'soil_erosion',
    # 'HWSD_gravel',
    'static_variables_category': [],  # 'landcover', 'lithology'
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        'climate':['MSWEP_P', 'MSWX_PET', 'MSWX_Tmean', 'MSWX_SWd'],
        'topography': ['GMTED_elevation', 'GMTED_slope', 'GMTED_aspectcosine', 'soil_depth'],  #
        'soil': ['HWSD_clay', 'HWSD_sand', 'HWSD_silt'],
        'geology': ['Permeability_no_permafrost', 'carbonate_sedimentary_rocks_frac', 'porosity'],
        'vegetation': ['MODIS_NDVI', 'grassland_fraction_2018', 'forest_fraction_1992', 'forest_fraction_2018', ],
    },
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['catchsize', 'population_density_2000_buffer', 'gdp_2000_buffer', 'intact_forest'],
    'negative_value_variables': ["Tmax", "Tmin", "MSWX_Tmean", "meanTa", "permeability", "permeability_permafrost",
                                 "Permeability_no_permafrost", "GMTED_aspectcosine", "GMTED_elevation", "MODIS_NDVI"],
}

tmp['station_ids'] = xr.open_dataset(tmp['input_nc_file']).station_ids.values

config_dataset_global_3434_3248_additional_1952 = SimpleNamespace(**tmp)
