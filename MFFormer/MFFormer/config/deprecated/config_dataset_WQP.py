import os
import pandas as pd
import xarray as xr
from types import SimpleNamespace
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

"""
'PRCP_nldas_extended', 'SRAD_nldas_extended', 'Tmax_nldas_extended','Tmin_nldas_extended', 'Vp_nldas_extended',
'prcp_maurer_extended','srad_maurer_extended', 'tmax_maurer_extended', 'tmin_maurer_extended','vp_maurer_extended', 
'prcp_daymet', 'srad_daymet', 'tmax_daymet', 'tmin_daymet','vp_daymet', 
'QObs'

"""
tmp = {
    'input_nc_file': os.path.join(server_dir, 'format_data/26.pretrain_data_processing/01.insitu_and_basin_average/07.WQP/WQP.nc'),
    'train_date_list': ["1980-01-01", "1999-12-31"],
    'val_date_list': ["2000-01-01", "2009-12-31"],
    'test_date_list': ["2010-01-01", "2020-12-31"],

    'time_series_variables': ["P_MSWEP", "SWd", "Pres", "SpecHum", "Tmax", "Tmin", "NO3", "PO4", "TP"],
    # "P","Lwd", "PET","RelHum", "Temp", "Wind",  "SSD"
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
    'mask_skip_variables': [],
    'negative_value_variables': ["Tmax", "Tmin", "Temp", "meanTa", "permeability", "permeability_permafrost",
                                 "Permeability_no_permafrost", "aspectcosine", "meanelevation"],

}

ds = xr.open_dataset(tmp['input_nc_file'])
station_ids = ds['station_ids'].values
indices_valid_NO3 = ds['mask_NO3'].values == 1
indices_valid_PO4 = ds['mask_PO4'].values == 1
indices_valid_SSD = ds['mask_SSD'].values == 1
indices_valid_TP = ds['mask_TP'].values == 1

indices_valid = indices_valid_NO3 | indices_valid_PO4 | indices_valid_TP
tmp['station_ids'] = station_ids[indices_valid]

config_dataset_WQP = SimpleNamespace(**tmp)
