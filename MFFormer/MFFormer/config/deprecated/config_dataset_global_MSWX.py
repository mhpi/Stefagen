"""
too much NaN: 'irrig',
error in data: 'aridity', 'permafrost',
repeat: 'SoilGrids1km_sand', 'SoilGrids1km_clay', 'SoilGrids1km_silt', 'permeability', 'water_cover', 'glaciers', 'permeability_permafrost',
'group_mask_dict': {
        'soil': ['HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt', 'soil_depth'],
        'geology': ['Permeability_no_permafrost', 'porosity'],
    },
    'soil_moisture',
"""
import os
from types import SimpleNamespace
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

tmp = {
    'input_nc_file': os.path.join(server_dir, 'format_data/26.pretrain_data_processing/02.satellite/01.MSWX/GLOBAL_GRID_MSWX_V10_20100101_20201231.nc'),
    'train_date_list': ["2010-01-01", "2010-12-31"],
    'val_date_list': ["2010-01-01", "2010-12-31"],
    'test_date_list': ["2010-01-01", "2010-12-31"],

    'time_series_variables': ['P_MSWEP', 'SWd', 'Pres', 'SpecHum', 'Tmax', 'Tmin', 'Wind'],
    'target_variables': [],
    'static_variables': ['meanP', 'seasonality_P', 'seasonality_PET', 'snow_fraction', 'snowfall_fraction',
                         'meanTa', 'NDVI', 'meanelevation', 'meanslope', 'HWSD_clay', 'HWSD_gravel', 'HWSD_sand',
                         'HWSD_silt', 'soil_depth', 'permeability_permafrost', 'porosity',
                         'carbonate_sedimentary_rocks'],  # 'lithology','landcover','ETPOT_Hargr',
    'static_variables_category': [],  # 'landcover', 'lithology'
    'station_ids': None,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': {
        'topography': ['meanelevation', 'meanslope'],
        'soil': ['HWSD_clay', 'HWSD_gravel', 'HWSD_sand', 'HWSD_silt', 'soil_depth'],
        'geology': ['permeability_permafrost', 'porosity'],
    },
    'data_type': 'grid',
    'sampling_interval': None,
    'mask_all_variables': [],
    'mask_skip_variables': [],
    'negative_value_variables': ["Tmax", "Tmin", "Temp", "meanTa", "permeability", "permeability_permafrost",
                                 "Permeability_no_permafrost", "aspectcosine", "meanelevation"],

}

config_dataset_global_grid_MSWEX = SimpleNamespace(**tmp)
