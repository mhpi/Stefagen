from types import SimpleNamespace
import os
import MFFormer
import pandas as pd
from MFFormer.datasets.load_path import LoadPath

server_dir = LoadPath.server()

camels_chem_basin_file = os.path.join(os.path.split(MFFormer.__file__)[0], "datasets/camels/camels_chem_516_basin_list.txt")
camels_chem_station_ids = pd.read_csv(camels_chem_basin_file, header=None, dtype=str).values[:, 0].tolist()

tmp = {
    'input_nc_file': os.path.join(server_dir, 'format_data/26.pretrain_data_processing/01.insitu_and_basin_average/06.CAMELS_Chem/CAMELS_Chem_daily.nc'),
    'train_date_list': ["1995-01-01", "2004-12-31"],
    'val_date_list': ["1980-10-01", "1989-09-30"],
    'test_date_list': ["2005-01-01", "2014-12-31"],

    'time_series_variables': ['PRCP_nldas_extended', 'SRAD_nldas_extended', 'Tmax_nldas_extended',
                              'Tmin_nldas_extended', 'Vp_nldas_extended'],
    # 'prcp_maurer_extended',
    # 'srad_maurer_extended', 'tmax_maurer_extended', 'tmin_maurer_extended',
    # 'vp_maurer_extended', 'prcp_daymet', 'srad_daymet', 'tmax_daymet', 'tmin_daymet',
    # 'vp_daymet', 'QObs'],  #
    'target_variables': ['Water_Temperature'], # Total_Dissolved_Nitrogen, Water_Temperature, Dissolved_Oxygen
    'static_variables': ['elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max', 'lai_diff', 'gvf_max',
                         'gvf_diff', 'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
                         'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'carbonate_rocks_frac',
                         'geol_permeability', 'p_mean', 'pet_mean', 'aridity', 'frac_snow', 'high_prec_freq',
                         'high_prec_dur', 'low_prec_freq', 'low_prec_dur'],
    'static_variables_category': [],
    'station_ids': camels_chem_station_ids,
    'regions': [],
    'add_coords': False,
    'group_mask_dict': None,
    'data_type': 'basin',
    'mask_all_variables': [],
    'mask_skip_variables': ['area_gages2'],
}
config_dataset_camels_chem = SimpleNamespace(**tmp)
