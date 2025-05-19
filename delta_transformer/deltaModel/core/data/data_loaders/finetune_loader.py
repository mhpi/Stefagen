import logging
import os
import json
import pickle
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple, List
from core.data.data_loaders.base import BaseDataLoader
from core.data.data_loaders.load_nc import NetCDFDataset
from core.utils.transform import cal_statistics
from core.data import intersect
import pandas as pd
from sklearn.exceptions import DataDimensionalityWarning

log = logging.getLogger(__name__)

class FineTuneDataLoader(BaseDataLoader):
    def __init__(
        self,
        config: Dict[str, Any],
        test_split: Optional[bool] = False,
        overwrite: Optional[bool] = False,
        holdout_index: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.nc_tool = NetCDFDataset()
        self.config = config
        self.test_split = test_split
        self.overwrite = overwrite
        
        self.nn_attributes = config['dpl_model']['nn_model'].get('attributes', [])
        self.nn_forcings = config['dpl_model']['nn_model'].get('forcings', [])
        self.phy_attributes = config['dpl_model']['phy_model'].get('attributes', [])
        self.phy_forcings = config['dpl_model']['phy_model'].get('forcings', [])
        self.data_name = config['observations']['name']
        self.all_forcings = config['observations']['forcings_all']
        self.all_attributes = config['observations']['attributes_all']
        self.target = config['train']['target']
        

        self.log_norm_vars = config['dpl_model']['phy_model'].get('use_log_norm', [])

        self.device = config['device']
        self.dtype = torch.float32
        self.train_dataset = None
        self.eval_dataset = None
        self.dataset = None
        self.norm_stats = None
        self.out_path = os.path.join(
            config.get('out_path', 'results'),
            'normalization_statistics.json',
        )
        self.test_mode = config.get('test_mode', {})
        self.is_spatial_test = (self.test_mode and 
                            self.test_mode.get('type') == 'spatial')
        if holdout_index is not None:
            self.holdout_index = holdout_index
        elif self.is_spatial_test and 'current_holdout_index' in self.test_mode:
            self.holdout_index = self.test_mode['current_holdout_index']
        elif self.is_spatial_test and self.test_mode.get('holdout_indexs'):
            # Default to first index if not specified
            self.holdout_index = self.test_mode['holdout_indexs'][0]
        else:
            self.holdout_index = None
            # log.info(f"Spatial test mode: {extent} with holdout index {self.holdout_index}, {len(self.holdout_basins)} holdout basins")
        self.load_dataset()

    def load_dataset(self) -> None:
        """Load dataset with spatial testing support."""
        train_range = {
            'start': self.config['train']['start_time'],
            'end': self.config['train']['end_time']
        }
        test_range = {
            'start': self.config['test']['start_time'],
            'end': self.config['test']['end_time']
        }
        
        if self.is_spatial_test:
            # For spatial testing, we need to separate by basins
            # Load all data first
            #need to fix this but this works ['test']['start_time'] - ['train']['end_time']
            full_range = {
                'start': self.config['test']['start_time'],
                'end': self.config['train']['end_time']
            }
            full_data = self._preprocess_data(full_range)
            
            # Then split into train and test by station ID
            self.train_dataset, self.eval_dataset = self._split_by_basin(full_data)
        else:
            # Standard temporal split
            if self.test_split:
                self.train_dataset = self._preprocess_data(train_range)
                self.eval_dataset = self._preprocess_data(test_range)
            else:
                full_range = {
                    'start': self.config['train']['start_time'],
                    'end': self.config['test']['end_time']
                }
                self.dataset = self._preprocess_data(full_range)

    def _split_by_basin(self, dataset):
        """Simple function to split dataset by HUC regions or PUB IDs."""
        if not dataset:
            return None, None
        
        try:
            extent = self.test_mode.get('extent')
            holdout_gages = []
            
            if extent == 'PUR':
                # Get the HUC regions to hold out
                huc_regions = self.test_mode.get('huc_regions', [])
                if not huc_regions or self.holdout_index >= len(huc_regions):
                    log.warning(f"Invalid holdout index: {self.holdout_index}")
                    return None, None
                    
                # Get the specific HUC regions for this holdout index
                holdout_hucs = huc_regions[self.holdout_index]
                log.info(f"Holding out basins from HUC regions: {holdout_hucs}")
                
                # Load the gage info file with HUC mappings
                gage_file = self.test_mode.get('gage_split_file')

                # Read the gage info CSV
                gageinfo = pd.read_csv(gage_file, dtype={"huc": int, "gage": str})
                
                # Get the basin IDs for the holdout HUCs
                holdout_hucs_int = [int(huc) for huc in holdout_hucs]
                holdout_gages = gageinfo[gageinfo['huc'].isin(holdout_hucs_int)]['gage'].tolist()
                
                log.info(f"Found {len(holdout_gages)} holdout basins from HUC regions {holdout_hucs}")
                # log.info(f"hold_out {holdout_gages}")
                
            elif extent == 'PUB':
                # Get the PUB IDs to hold out
                pub_ids = self.test_mode.get('PUB_ids', [])
                if not pub_ids or self.holdout_index >= len(pub_ids):
                    log.warning(f"Invalid holdout index: {self.holdout_index}")
                    return None, None
                    
                # Get the specific PUB ID for this holdout index
                holdout_pub = pub_ids[self.holdout_index]
                log.info(f"Holding out basins from PUB ID: {holdout_pub}")
                
                # Load the gage info file with PUB mappings
                gage_file = self.test_mode.get('gage_split_file')
                if not os.path.exists(gage_file):
                    log.error(f"Gage file not found: {gage_file}")
                    return None, None
                    
                # Read the gage info CSV
                gageinfo = pd.read_csv(gage_file, dtype={"PUB_ID": int, "gage": str})
                
                # Get the basin IDs for the holdout PUB ID
                holdout_gages = gageinfo[gageinfo['PUB_ID'] == holdout_pub]['gage'].tolist()
                
                log.info(f"Found {len(holdout_gages)} holdout basins from PUB ID {holdout_pub}")
                
            else:
                log.error(f"Unknown extent: {extent}")
                return None, None
            
            # Load the subset file which contains our active basin list
            subset_path = self.config['observations']['subset_path']
            if not os.path.exists(subset_path):
                log.error(f"Subset file not found: {subset_path}")
                return None, None

            with open(subset_path, 'r') as f:
                content = f.read().strip()
                # Handle Python list format
                if content.startswith('[') and content.endswith(']'):
                    content = content.strip('[]')
                    all_basins = [item.strip().strip(',') for item in content.split() if item.strip().strip(',')]
                else:
                    all_basins = [line.strip() for line in content.split() if line.strip()]
            
            log.info(f"Parsed {len(all_basins)} basins from subset file")
            holdout_gages_int = set()
            for basin in holdout_gages:
                basin_str = str(basin).strip()
                holdout_gages_int.add(int(basin_str))


            test_indices = []
            train_indices = []
            for i, basin in enumerate(all_basins):
                basin_int = int(str(basin).strip())
                if basin_int in holdout_gages_int:
                    test_indices.append(i)
                else:
                    train_indices.append(i)


            
            # Verify we have test basins
            if not test_indices:
                raise ValueError("No test basins found! Check your region settings and basin IDs.")
                
            # Now split the dataset using these indices
            train_data = {}
            test_data = {}
            
            # Create index tensors
            train_indices_tensor = torch.tensor(train_indices, device='cpu')
            test_indices_tensor = torch.tensor(test_indices, device='cpu')
            
            for key, tensor in dataset.items():
                if tensor is None:
                    continue
                    
                # Move tensor to CPU for safe indexing
                cpu_tensor = tensor.to('cpu')
                
                # Handle different tensor shapes
                if len(cpu_tensor.shape) == 3:
                    if cpu_tensor.shape[0] == len(all_basins):  # [basins, time, features]
                        train_data[key] = cpu_tensor[train_indices_tensor]
                        test_data[key] = cpu_tensor[test_indices_tensor]
                    else:  # [time, basins, features] for x_phy and target
                        train_data[key] = cpu_tensor[:, train_indices_tensor]
                        test_data[key] = cpu_tensor[:, test_indices_tensor]
                elif len(cpu_tensor.shape) == 2:  # [basins, features]
                    train_data[key] = cpu_tensor[train_indices_tensor]
                    test_data[key] = cpu_tensor[test_indices_tensor]
                else:
                    # Just copy for unusual shapes
                    train_data[key] = tensor
                    test_data[key] = tensor
                
                # Move back to original device
                train_data[key] = train_data[key].to(tensor.device)
                test_data[key] = test_data[key].to(tensor.device)
            
            return train_data, test_data
        
        except Exception as e:
            log.error(f"Error splitting dataset by basin: {e}")
            return None, None
                
    def _preprocess_data(self, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Load and preprocess both neural network and physics data."""
        scope = 'train' if t_range['end'] == self.config['train']['end_time'] else 'test'
        log.info(f"Starting preprocessing for {scope} data with time range: {t_range}")
        
        try:
            # First load neural network data
            log.info("Loading neural network data...")
            nn_data = self._load_nn_data(scope, t_range)
            log.info(f"Neural network data shapes - x_nn: {nn_data['x_nn'].shape}, c_nn: {nn_data['c_nn'].shape}, xc_nn_norm: {nn_data['xc_nn_norm'].shape}")
            
            # Load physics model data
            if scope == 'train':
                data_path = self.config['observations']['train_path']
            else:
                data_path = self.config['observations']['test_path']
            # log.info(f"Loading physics data from: {data_path}")
                
            # Load the pickle file
            with open(data_path, 'rb') as f:
                forcings, target, attributes = pickle.load(f)
            log.info(f"Loaded raw data shapes - forcings: {forcings.shape}, target: {target.shape}, attributes: {attributes.shape}")
                
            # Convert dates to indices for time slicing
            start_date = pd.to_datetime(t_range['start'].replace('/', '-'))
            end_date = pd.to_datetime(t_range['end'].replace('/', '-'))
            all_dates = pd.date_range(
                self.config['observations']['start_time'].replace('/', '-'),
                self.config['observations']['end_time'].replace('/', '-'),
                freq='D'
            )
            idx_start = all_dates.get_loc(start_date)
            idx_end = all_dates.get_loc(end_date) + 1
            # log.info(f"Time slice indices - start: {idx_start}, end: {idx_end}")
            
            # Process forcings with correct transposition order
            forcings = np.transpose(forcings[:, idx_start:idx_end], (1, 0, 2))
            # log.info(f"Transposed forcings shape: {forcings.shape}")
            
            # Get physics model indices
            phy_forc_idx = []
            for name in self.phy_forcings:
                if name not in self.all_forcings:
                    log.warning(f"Physics forcing {name} not found in available forcings: {self.all_forcings}")
                    raise ValueError(f"Forcing {name} not listed in available forcings.")
                phy_forc_idx.append(self.all_forcings.index(name))
            
            # Process attributes for physics model
            # if not self.phy_attributes:
            #     # If no physics attributes specified, use all attributes
            #     c_phy = attributes[:, 0]   # Use all attributes
            #     log.info(f"No physics attributes specified")
            # else:
            phy_attr_idx = []
            for attr in self.phy_attributes:
                if attr not in self.all_attributes:
                    log.warning(f"Physics attribute {attr} not found in available attributes: {self.all_attributes}")
                    raise ValueError(f"Attribute {attr} not in available attributes")
                phy_attr_idx.append(self.all_attributes.index(attr))
            c_phy = attributes[:, phy_attr_idx]
            
            x_phy = forcings[:, :, phy_forc_idx]
            # log.info(f"Physics model data shapes - x_phy: {x_phy.shape}, c_phy: {c_phy.shape}")
            
            # Process target data with proper transposition
            target = np.transpose(target[:, idx_start:idx_end], (1, 0, 2))
            # log.info(f"Transposed target shape: {target.shape}")
            
            # Apply subsetting if needed
            if self.data_name.split('_')[-1] != '671':
                subset_path = self.config['observations']['subset_path']
                gage_id_path = self.config['observations']['gage_info']
                # log.info(f"Applying subsetting using paths - subset: {subset_path}, gage_info: {gage_id_path}")
                
                with open(subset_path, 'r') as f:
                    selected_basins = json.load(f)
                gage_info = np.load(gage_id_path)
                subset_idx = []
                station_to_idx = {str(id): i for i, id in enumerate(gage_info)}
                for station_id in selected_basins:
                    if str(station_id) in station_to_idx:
                        subset_idx.append(station_to_idx[str(station_id)])
                # log.info(f"Number of basins after subsetting: {len(subset_idx)}")
                
                x_phy = x_phy[:, subset_idx, :]
                c_phy = c_phy[subset_idx,:] if c_phy.size > 0 else c_phy
                x_nn = nn_data['x_nn'][:, :, :]  # Already subset during loading
                c_nn = nn_data['c_nn'][:, :]     # Already subset during loading
                target = target[:, subset_idx, :]
                log.info(f"Data shapes after subsetting - x_phy: {x_phy.shape}, c_phy: {c_phy.shape}, target: {target.shape}")
            else:
                x_nn = nn_data['x_nn']
                c_nn = nn_data['c_nn']

            # Convert flow to mm/day if necessary (following HydroDataLoader)
            target = self._flow_conversion(c_nn, target)
            
            # Log NaN statistics for target data
            nan_count = np.isnan(target).sum()
            if nan_count > 0:
                nan_percent = (nan_count / target.size) * 100
                log.warning(f"Target contains {nan_count} NaN values ({nan_percent:.2f}%)")
            
            # Normalize data using HydroDataLoader approach
            self.load_norm_stats(x_nn, c_nn, target)
            xc_nn_norm = self.normalize(x_nn, c_nn)

            # Debug and additional logging
            # log.info(f"Target data before tensor conversion - has NaNs: {np.isnan(target).any()}")
            
            # Convert to tensors and combine datasets
            dataset = {
                'x_nn': nn_data['x_nn'],        # Already a tensor
                'c_nn': nn_data['c_nn'],        # Already a tensor
                'xc_nn_norm': nn_data['xc_nn_norm'], # Already a tensor
                'x_phy': self.to_tensor(x_phy),
                'c_phy': self.to_tensor(c_phy),
                'target': self.to_tensor(target)
            }

            # Final check for NaNs in tensors
            # for key, value in dataset.items():
            #     if torch.is_tensor(value):
            #         has_nan = torch.isnan(value).any().item()
            #         log.info(f"Final tensor shape for {key}: {value.shape}, dtype: {value.dtype}, device: {value.device}, has NaNs: {has_nan}")
            #         if has_nan and key == 'target':
            #             log.warning(f"Replacing NaNs in {key} tensor with zeros")
            #             dataset[key] = torch.nan_to_num(value, nan=0.0)
            
            return dataset
                
        except Exception as e:
            log.error(f"Error in data preprocessing: {str(e)}")
            raise

    def _flow_conversion(self, c_nn: torch.Tensor, target: np.ndarray) -> np.ndarray:
        """Convert hydraulic flow from ft3/s to mm/day, following HydroDataLoader."""
        for name in ['flow_sim', 'streamflow', 'sf']:
            if name in self.target:
                target_index = self.target.index(name)
                target_temp = target[:, :, target_index].copy()
                
                try:
                    area_name = self.config['observations']['area_name']
                    # Handle if c_nn is already tensor
                    if torch.is_tensor(c_nn):
                        c_nn_np = c_nn.cpu().numpy()
                    else:
                        c_nn_np = c_nn
                        
                    # Get basin area
                    basin_area = c_nn_np[:, self.nn_attributes.index(area_name)]
                    basin_area = basin_area.reshape(-1, 1)  # Reshape to column
                    
                    # Expand area to match target shape
                    area = np.tile(basin_area, (1, target_temp.shape[0])).T
                    
                    # Convert flow to mm/day
                    converted_flow = ((10 ** 3) * target_temp * 0.0283168 * 3600 * 24 / 
                                     (area * (10 ** 6)))
                    
                    # Update target with converted flow
                    target[:, :, target_index] = converted_flow
                    
                    # Log conversion info
                    # log.info(f"Converted flow data for {name}, shape: {converted_flow.shape}")
                    if np.isnan(converted_flow).any():
                        nan_count = np.isnan(converted_flow).sum()
                        log.warning(f"Flow conversion introduced {nan_count} NaN values")
                        
                except (KeyError, ValueError) as e:
                    log.warning(f"Could not convert flow units: {e}")
                    # Continue with unconverted flow
        
        return target
        
    def load_norm_stats(
        self,
        x_nn: torch.Tensor, 
        c_nn: torch.Tensor, 
        target: np.ndarray
    ) -> None:
        """Load or calculate normalization statistics, following HydroDataLoader."""
        if os.path.isfile(self.out_path) and not self.overwrite:
            if not self.norm_stats:
                try:
                    with open(self.out_path, 'r') as f:
                        self.norm_stats = json.load(f)
                    # log.info(f"Loaded normalization statistics from {self.out_path}")
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    log.warning(f"Could not load norm stats: {e}")
                    self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)
        else:
            # Init normalization stats if file doesn't exist or overwrite is True
            self.norm_stats = self._init_norm_stats(x_nn, c_nn, target)
    
    def _init_norm_stats(
        self,
        x_nn: torch.Tensor,
        c_nn: torch.Tensor,
        target: np.ndarray
    ) -> Dict[str, List[float]]:
        """Calculate normalization statistics, following HydroDataLoader."""
        stat_dict = {}
        
        # Convert tensors to numpy if needed
        if torch.is_tensor(x_nn):
            x_nn_np = x_nn.cpu().numpy()
        else:
            x_nn_np = x_nn
            
        if torch.is_tensor(c_nn):
            c_nn_np = c_nn.cpu().numpy()
        else:
            c_nn_np = c_nn
        
        # Get basin areas from attributes
        basin_area = self._get_basin_area(c_nn_np)

        # Forcing variable stats
        for k, var in enumerate(self.nn_forcings):
            try:
                if var in self.log_norm_vars:
                    stat_dict[var] = self._calc_gamma_stats(x_nn_np[:, :, k])
                else:
                    stat_dict[var] = self._calc_norm_stats(x_nn_np[:, :, k])
            except Exception as e:
                log.warning(f"Error calculating stats for {var}: {e}")
                stat_dict[var] = [0, 1, 0, 1]  # Default values

        # Attribute variable stats
        for k, var in enumerate(self.nn_attributes):
            try:
                stat_dict[var] = self._calc_norm_stats(c_nn_np[:, k])
            except Exception as e:
                log.warning(f"Error calculating stats for {var}: {e}")
                stat_dict[var] = [0, 1, 0, 1]  # Default values

        # Target variable stats
        for i, name in enumerate(self.target):
            try:
                if name in ['flow_sim', 'streamflow', 'sf']:
                    stat_dict[name] = self._calc_norm_stats(
                        np.swapaxes(target[:, :, i:i+1], 1, 0).copy(),
                        basin_area,
                    )
                else:
                    stat_dict[name] = self._calc_norm_stats(
                        np.swapaxes(target[:, :, i:i+1], 1, 0),
                    )
            except Exception as e:
                log.warning(f"Error calculating stats for {name}: {e}")
                stat_dict[name] = [0, 1, 0, 1]  # Default values

        # Save statistics to file
        try:
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
            with open(self.out_path, 'w') as f:
                json.dump(stat_dict, f, indent=4)
            # log.info(f"Saved normalization statistics to {self.out_path}")
        except Exception as e:
            log.warning(f"Could not save norm stats: {e}")
        
        return stat_dict

    def _calc_norm_stats(
        self,
        x: np.ndarray, 
        basin_area: np.ndarray = None, 
    ) -> List[float]:
        """
        Calculate statistics for normalization with optional basin area adjustment.
        Follows HydroDataLoader implementation.
        """
        # Handle invalid values
        x = x.copy()  # Create a copy to avoid modifying original data
        x[x == -999] = np.nan
        if basin_area is not None:
            x[x < 0] = 0  # Specific to basin normalization

        # Basin area normalization
        if basin_area is not None:
            nd = len(x.shape)
            if nd == 3 and x.shape[2] == 1:
                x = x[:, :, 0]  # Unsqueeze the original 3D matrix
            temparea = np.tile(basin_area, (1, x.shape[1]))
            flow = (x * 0.0283168 * 3600 * 24) / (temparea * (10 ** 6)) * 10 ** 3
            x = flow  # Replace x with flow for further calculations

        # Flatten and exclude NaNs and invalid values
        a = x.flatten()
        if basin_area is None:
            a = np.swapaxes(x, 1, 0).flatten() if len(x.shape) > 1 else x.flatten()
        b = a[(~np.isnan(a)) & (a != -999999)]
        if b.size == 0:
            log.warning("No valid values for statistics calculation, using defaults")
            b = np.array([0])

        # Calculate statistics
        transformed = np.log10(np.sqrt(b) + 0.1) if basin_area is not None else b
        p10, p90 = np.percentile(transformed, [10, 90]).astype(float)
        mean = np.mean(transformed).astype(float)
        std = np.std(transformed).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _calc_gamma_stats(self, x: np.ndarray) -> List[float]:
        """
        Calculate gamma statistics for streamflow and precipitation data.
        Follows HydroDataLoader implementation.
        """
        a = np.swapaxes(x, 1, 0).flatten()
        b = a[(~np.isnan(a))]
        
        if b.size == 0:
            log.warning("No valid values for gamma statistics calculation, using defaults")
            return [0, 1, 0, 1]
        
        b = np.log10(np.sqrt(b) + 0.1)

        p10, p90 = np.percentile(b, [10, 90]).astype(float)
        mean = np.mean(b).astype(float)
        std = np.std(b).astype(float)

        return [p10, p90, mean, max(std, 0.001)]
    
    def _get_basin_area(self, c_nn: np.ndarray) -> np.ndarray:
        """Get basin area from attributes, following HydroDataLoader."""
        try:
            area_name = self.config['observations']['area_name']
            basin_area = c_nn[:, self.nn_attributes.index(area_name)][:, np.newaxis]
            return basin_area
        except (KeyError, ValueError) as e:
            log.warning(f"No area information found: {e}. Basin area norm will not be applied.")
            return None

    def normalize(self, x_nn: torch.Tensor, c_nn: torch.Tensor) -> np.ndarray:
        """
        Normalize data for neural network with robust handling of statistics.
        """
        # Convert tensors to numpy if needed
        if torch.is_tensor(x_nn):
            x_nn_np = x_nn.cpu().numpy()
        else:
            x_nn_np = x_nn
            
        if torch.is_tensor(c_nn):
            c_nn_np = c_nn.cpu().numpy()
        else:
            c_nn_np = c_nn
        
        # Calculate basic statistics if needed
        # This handles the case where norm_stats is loaded but has the wrong keys
        if self.norm_stats is None or "x_mean" not in self.norm_stats:
            log.warning("Calculating normalization statistics directly (missing x_mean)")
            # Calculate basic statistics
            epsilon = 1e-5
            
            # Handle time series data
            x_mean = np.nanmean(x_nn_np, axis=(0, 1), keepdims=True)
            x_std = np.nanstd(x_nn_np, axis=(0, 1), keepdims=True)
            x_std = np.where(x_std < epsilon, 1.0, x_std)
            
            # Handle static data
            c_mean = np.nanmean(c_nn_np, axis=0, keepdims=True)
            c_std = np.nanstd(c_nn_np, axis=0, keepdims=True)
            c_std = np.where(c_std < epsilon, 1.0, c_std)
            
            # Create or update norm_stats
            if self.norm_stats is None:
                self.norm_stats = {}
                
            self.norm_stats["x_mean"] = x_mean
            self.norm_stats["x_std"] = x_std
            self.norm_stats["c_mean"] = c_mean
            self.norm_stats["c_std"] = c_std
        
        # Apply normalization with type checking
        epsilon = 1e-5
        
        # Normalize time series data - maintaining shape (stations, time, features)
        x_nn_norm = ((x_nn_np - self.norm_stats["x_mean"]) / 
                    (self.norm_stats["x_std"] + epsilon)).astype(np.float32)
        
        # Normalize static data - shape (stations, features)
        c_nn_norm = ((c_nn_np - self.norm_stats["c_mean"]) / 
                    (self.norm_stats["c_std"] + epsilon)).astype(np.float32)
        
        # Replace invalid values
        x_nn_norm = np.where(np.isfinite(x_nn_norm), 
                            x_nn_norm, 0).astype(np.float32)
        c_nn_norm = np.where(np.isfinite(c_nn_norm), 
                            c_nn_norm, 0).astype(np.float32)
                                
        # Expand static data to match time series dimensions
        # [stations, features] -> [stations, time, features]
        c_nn_norm_expanded = np.repeat(
            c_nn_norm[:, np.newaxis, :],  # Add time dimension: [stations, 1, features]
            x_nn_norm.shape[1],      # Repeat for each timestep
            axis=1                          # Along the time dimension
        )
        
        log.info(f"Normalization shapes - x_nn_norm: {x_nn_norm.shape}, c_nn_norm_expanded: {c_nn_norm_expanded.shape}, xc_nn_norm: {x_nn_norm.shape[0], x_nn_norm.shape[1], x_nn_norm.shape[2] + c_nn_norm_expanded.shape[2]}")
        
        # Concatenate along features dimension (axis=2)
        # Both inputs are now shape (stations, time, features)
        xc_nn_norm = np.concatenate((x_nn_norm, c_nn_norm_expanded), axis=2)
        
        return xc_nn_norm
    def _to_norm(self, data: np.ndarray, vars: List[str]) -> np.ndarray:
        """Standard data normalization, following HydroDataLoader."""
        if not self.norm_stats:
            log.warning("No normalization statistics available, using identity normalization")
            return data
            
        data_norm = np.zeros(data.shape)

        for k, var in enumerate(vars):
            if var not in self.norm_stats:
                log.warning(f"No normalization stats for {var}, skipping")
                continue
                
            stat = self.norm_stats[var]

            try:
                if len(data.shape) == 3:
                    if var in self.log_norm_vars:
                        data[:, :, k] = np.log10(np.sqrt(np.maximum(data[:, :, k], 0)) + 0.1)
                    data_norm[:, :, k] = (data[:, :, k] - stat[2]) / stat[3]
                elif len(data.shape) == 2:
                    if var in self.log_norm_vars:
                        data[:, k] = np.log10(np.sqrt(np.maximum(data[:, k], 0)) + 0.1)
                    data_norm[:, k] = (data[:, k] - stat[2]) / stat[3]
                else:
                    raise DataDimensionalityWarning("Data dimension must be 2 or 3.")
            except Exception as e:
                log.warning(f"Error normalizing {var}: {e}")
                # Copy original data if normalization fails
                if len(data.shape) == 3:
                    data_norm[:, :, k] = data[:, :, k]
                else:
                    data_norm[:, k] = data[:, k]
            
        if len(data_norm.shape) < 3:
            return data_norm
        else:
            return np.swapaxes(data_norm, 1, 0)

    def _from_norm(self, data_norm: np.ndarray, vars: List[str]) -> np.ndarray:
        """De-normalize data, following HydroDataLoader."""
        if not self.norm_stats:
            log.warning("No normalization statistics available, using identity denormalization")
            return data_norm
            
        data = np.zeros(data_norm.shape)
                
        for k, var in enumerate(vars):
            if var not in self.norm_stats:
                log.warning(f"No normalization stats for {var}, skipping")
                continue
                
            stat = self.norm_stats[var]
            
            try:
                if len(data_norm.shape) == 3:
                    data[:, :, k] = data_norm[:, :, k] * stat[3] + stat[2]
                    if var in self.log_norm_vars:
                        data[:, :, k] = (np.power(10, data[:, :, k]) - 0.1) ** 2
                elif len(data_norm.shape) == 2:
                    data[:, k] = data_norm[:, k] * stat[3] + stat[2]
                    if var in self.log_norm_vars:
                        data[:, k] = (np.power(10, data[:, k]) - 0.1) ** 2
                else:
                    raise DataDimensionalityWarning("Data dimension must be 2 or 3.")
            except Exception as e:
                log.warning(f"Error denormalizing {var}: {e}")
                # Copy normalized data if denormalization fails
                if len(data_norm.shape) == 3:
                    data[:, :, k] = data_norm[:, :, k]
                else:
                    data[:, k] = data_norm[:, k]

        if len(data.shape) < 3:
            return data
        else:
            return np.swapaxes(data, 1, 0)
    def _load_nn_data(self, scope: str, t_range: Dict[str, str]) -> Dict[str, torch.Tensor]:
        """Load and process neural network data from NetCDF."""
        time_range = [t_range['start'].replace('/', '-'), t_range['end'].replace('/', '-')]
        warmup_days = self.config['dpl_model']['phy_model']['warm_up']

        try:
            # Handle station IDs
            try:
                # First check if we need to use a subset of stations
                if self.data_name.split('_')[-1] != '671':
                    subset_path = self.config['observations']['subset_path']
                    gage_id_path = self.config['observations']['gage_info']
                    
                    # Load the subset information
                    with open(subset_path, 'r') as f:
                        selected_basins = json.load(f)
                        
                    log.info(f"Loaded {len(selected_basins)} stations from subset file")
                    
                    # Convert to string if needed
                    station_ids = [str(id) for id in selected_basins] 
                    
                    # Important: Instead of trying to filter stations during NetCDF loading,
                    # we'll load all stations and filter afterward
                    time_series_data, static_data, date_range = self.nc_tool.nc2array(
                        self.config['data_path'],
                        station_ids=None,  # Load all stations
                        time_range=time_range,
                        time_series_variables=self.nn_forcings,
                        static_variables=self.nn_attributes,
                        add_coords=True,
                        warmup_days=warmup_days
                    )
                    
                    # Now load the station IDs from the NetCDF file to use for filtering
                    gage_info = np.load(gage_id_path)
                    
                    # Create a mapping from station ID to index
                    station_to_idx = {}
                    for i, station_id in enumerate(gage_info):
                        station_to_idx[str(station_id)] = i
                    
                    # Get indices of selected stations that exist in the dataset
                    subset_idx = []
                    for station_id in station_ids:
                        if station_id in station_to_idx:
                            subset_idx.append(station_to_idx[station_id])
                    
                    if not subset_idx:
                        raise ValueError(f"No matching stations found between subset and dataset")
                    
                    log.info(f"Filtering data to {len(subset_idx)} stations")
                    
                    # Filter the data to include only the selected stations
                    time_series_data = time_series_data[subset_idx]
                    static_data = static_data[subset_idx]
                else:
                    # No subsetting needed, load all stations
                    time_series_data, static_data, date_range = self.nc_tool.nc2array(
                        self.config['data_path'],
                        station_ids=None,
                        time_range=time_range,
                        time_series_variables=self.nn_forcings,
                        static_variables=self.nn_attributes,
                        add_coords=True,
                        warmup_days=warmup_days
                    )
            except Exception as e:
                log.error(f"Error loading neural network data: {str(e)}")
                raise
            
            # Handle coordinates
            if static_data.shape[1] >= 2:  # Make sure we have enough columns
                lon = static_data[:, -1]
                lat = static_data[:, -2]
                static_data = static_data[:, :-2]
            
            # Calculate normalization statistics if needed
            if self.norm_stats is None:
                # Ensure data is in float format
                time_series_float = time_series_data.astype(np.float32)
                static_float = static_data.astype(np.float32)
                
                # Calculate basic statistics
                epsilon = 1e-5
                
                # Handle time series data
                x_mean = np.nanmean(time_series_float, axis=(0, 1), keepdims=True)
                x_std = np.nanstd(time_series_float, axis=(0, 1), keepdims=True)
                x_std = np.where(x_std < epsilon, 1.0, x_std)
                
                # Handle static data
                c_mean = np.nanmean(static_float, axis=0, keepdims=True)
                c_std = np.nanstd(static_float, axis=0, keepdims=True)
                c_std = np.where(c_std < epsilon, 1.0, c_std)
                
                self.norm_stats = {
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "c_mean": c_mean,
                    "c_std": c_std
                }
            
            # Log dimension information
            log.info(f"Time series data shape: {time_series_data.shape}")  # (stations, time, features)
            log.info(f"Static data shape: {static_data.shape}")           # (stations, features)
            
            # Convert data to float32
            time_series_data = time_series_data.astype(np.float32)  # [stations, time, features]
            static_data = static_data.astype(np.float32)           # [stations, features]
            
            # Apply normalization
            epsilon = 1e-5
            
            # Normalize time series data - maintaining shape (stations, time, features)
            time_series_norm = ((time_series_data - self.norm_stats["x_mean"]) / 
                            (self.norm_stats["x_std"] + epsilon)).astype(np.float32)
            
            # Normalize static data - shape (stations, features)
            static_norm = ((static_data - self.norm_stats["c_mean"]) / 
                        (self.norm_stats["c_std"] + epsilon)).astype(np.float32)
            
            # Replace invalid values
            time_series_norm = np.where(np.isfinite(time_series_norm), 
                                    time_series_norm, 0).astype(np.float32)
            static_norm = np.where(np.isfinite(static_norm), 
                                static_norm, 0).astype(np.float32)
                                
            # Expand static data to match time series dimensions
            static_norm_expanded = np.repeat(
                static_norm[:, np.newaxis, :],  # Add time dimension: [stations, 1, features]
                time_series_norm.shape[1],      # Repeat for each timestep
                axis=1                          # Along the time dimension
            )
            
            # Concatenate along features dimension (axis=2)
            xc_nn_norm = np.concatenate((time_series_norm, static_norm_expanded), axis=2)
            
            return {
                'x_nn': self.to_tensor(time_series_data),      # (stations, time, features)
                'c_nn': self.to_tensor(static_data),           # (stations, features)
                'xc_nn_norm': self.to_tensor(xc_nn_norm)       # (stations, time, combined_features)
            }
            
        except Exception as e:
            log.error(f"Error loading neural network data: {str(e)}")
            raise
    