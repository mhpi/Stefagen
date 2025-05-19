import os
import pandas as pd
from types import SimpleNamespace
import xarray as xr
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()
camels_basin_file = os.path.join('/storage/home/nrk5343/work/30.MFFormer/MFFormer/datasets/camels/531_basin_list.txt')
camels_station_ids = pd.read_csv(camels_basin_file, header=None, dtype=str).values[:, 0].tolist()
tmp = {
    'input_nc_file': os.path.join("/storage/home/nrk5343/scratch/Data/GMD_Data.nc"),
    'train_date_list': ["2015-04-01", "2019-12-31"],
    'val_date_list': ["2020-01-01", "2020-06-30"],
    'test_date_list': ["2020-07-01", "2020-12-31"],
    
    'time_series_variables': [
        'Albedo_BSA', 'Albedo_WSA', 'GPM', 'LST_Day', 'LST_Night', 
        'MSWEP', 'SMAP_9km_AM', 'forecast_albedo', 
        'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 
        'soil_temperature_level_1', 'surface_pressure', 
        'surface_solar_radiation_downwards', 'temperature_2m', 
        'total_evaporation', 'total_precipitation', 
        'u_component_of_wind_10m', 'v_component_of_wind_10m', 
        'volumetric_soil_water_layer_1'
    ],
    
    'target_variables': ['soil_moisture'],  
    
    'static_variables': [
        'lat', 'lon', 'slope', 'T_SAND', 'AVERAGE_LST_10km_LST_Day', 'Urban', 
        'landcover_ESA_2018', 'AVERAGE_Albedo_10km_Albedo_WSA', 
        'T_CLAY', 'AVERAGE_LST_10km_LST_Night', 'T_TEXTURE', 
        'Open_Water', 'AVERAGE_NDVI_p05', 'pcurv', 'T_BULK_DEN', 
        'AVERAGE_Albedo_10km_Albedo_BSA', 'aspectcosine', 
        'roughness', 'Snow_Ice', 'T_SILT', 'AVERAGE_SMAP_9km_AM', 
        'elevation'
    ],
    
    'static_variables_category': [],  # Add categorical variables here if any
    
    'station_ids': None,  # Will be loaded from the NC file
    
    'regions': [],  # Add regions if you want to filter by region
    
    'add_coords': False,  # Include lat/lon coordinates
    
    'group_mask_dict': {
        'topography': ['slope', 'pcurv', 'aspectcosine', 'roughness', 'elevation'],
        'soil': ['T_SAND', 'T_CLAY', 'T_TEXTURE', 'T_BULK_DEN', 'T_SILT'],
        'landcover': ['Urban', 'landcover_ESA_2018', 'Open_Water', 'Snow_Ice'],
        'remote_sensing': [
            'AVERAGE_LST_10km_LST_Day', 'AVERAGE_Albedo_10km_Albedo_WSA',
            'AVERAGE_LST_10km_LST_Night', 'AVERAGE_NDVI_p05',
            'AVERAGE_Albedo_10km_Albedo_BSA', 'AVERAGE_SMAP_9km_AM'
        ]
    },
    
    'data_type': 'station',  # Changed from 'basin' to 'station'
    
    'mask_all_variables': [],
    
    'mask_skip_variables': [],
    
    'negative_value_variables': ["temperature_2m", "LST_Day", "LST_Night"],  # Variables that can have negative values
}

# Load station IDs from the NetCDF file
ds = xr.open_dataset(tmp['input_nc_file'])
tmp['station_ids'] = ds.station_ids.values
# print(tmp['station_ids'])
# Close the dataset after reading
ds.close()

# Create configuration object
config_soil_moisture = SimpleNamespace(**tmp)