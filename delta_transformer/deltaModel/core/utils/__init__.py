import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dates import Dates

from core.utils.path_builder import PathBuilder

log = logging.getLogger(__name__)

def set_system_spec(cuda_devices: Optional[list] = None) -> Tuple[str, str]:
    """Set the device and data type for the model on user's system.

    Parameters
    ----------
    cuda_devices : list or int, optional
        CUDA device(s) to use. If None or empty list, uses available device or falls back to CPU.

    Returns
    -------
    Tuple[str, str]
        The device type and data type for the model.
    """
    try:
        if cuda_devices is not None and not isinstance(cuda_devices, list):
            # Handle single device ID case
            if torch.cuda.is_available() and cuda_devices < torch.cuda.device_count():
                device = torch.device(f'cuda:{cuda_devices}')
                torch.cuda.set_device(device)
                log.info(f"Using CUDA device {cuda_devices}: {torch.cuda.get_device_name(cuda_devices)}")
            else:
                log.warning(f"CUDA device {cuda_devices} not available, falling back to CPU")
                device = torch.device('cpu')
        
        elif cuda_devices and torch.cuda.is_available():
            # Handle multi-device case
            device = torch.device(f'cuda:{cuda_devices[0]}')
            torch.cuda.set_device(device)
            log.info(f"Using CUDA device {cuda_devices[0]}: {torch.cuda.get_device_name(cuda_devices[0])}")
            
        elif torch.cuda.is_available():
            # Use default CUDA device if available
            device = torch.device(f'cuda:{torch.cuda.current_device()}')
            torch.cuda.set_device(device)
            log.info(f"Using default CUDA device: {torch.cuda.get_device_name()}")
            
        elif torch.backends.mps.is_available():
            # Use Mac M-series ARM architecture if available
            device = torch.device('mps')
            log.info("Using MPS device")
            
        else:
            # Fall back to CPU
            device = torch.device('cpu')
            log.info("No GPU available, using CPU")
        
        dtype = torch.float32
        return str(device), str(dtype)
        
    except Exception as e:
        log.warning(f"Error setting up device: {str(e)}. Falling back to CPU")
        return 'cpu', str(torch.float32)


def set_randomseed(seed=0) -> None:
    """Fix random seeds for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed to set. If None, a random seed is used. Default is 0.
    """
    if seed == None:
        # seed = int(np.random.uniform(low=0, high=1e6))
        pass

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.use_deterministic_algorithms(True)
    except Exception as e:
        log.warning(f"Error fixing randomseed: {e}")


def initialize_config(config: Union[DictConfig, dict]) -> Dict[str, Any]:
    """Parse and initialize configuration settings.
    
    Parameters
    ----------
    config : DictConfig
        Configuration settings from Hydra.
        
    Returns
    -------
    dict
        Formatted configuration settings.
    """
    if type(config) == DictConfig:
        try:
            config = OmegaConf.to_container(config, resolve=True)
        except ValidationError as e:
            log.exception("Configuration validation error", exc_info=e)
            raise e

    if config.get('use_multi_gpu', False) and config.get('devices'):
        device_ids = [int(id.strip()) for id in config['devices'].split(',')]
        config['device'], config['dtype'] = set_system_spec(device_ids)
    else:
        config['device'], config['dtype'] = set_system_spec(config['gpu_id'])

    train_time = Dates(config['train'], config['dpl_model']['rho'])
    test_time = Dates(config['test'], config['dpl_model']['rho'])
    all_time = Dates(config['observations'], config['dpl_model']['rho'])

    config['train_time'] = [train_time.start_time, train_time.end_time]
    config['test_time'] = [test_time.start_time, test_time.end_time]
    config['experiment_time'] = [train_time.start_time, test_time.end_time]
    config['all_time'] = [all_time.start_time, all_time.end_time]   

    if 'test_mode' in config:
        if config['test_mode'].get('gage_split_file'):
            if not os.path.exists(config['test_mode']['gage_split_file']):
                log.warning(f"Gage split file not found: {config['test_mode']['gage_split_file']}")
                
        if config['test_mode'].get('type') == 'spatial':
            extent = config['test_mode'].get('extent')
            holdout_indexs = config['test_mode'].get('holdout_indexs', [])
            
            if extent == 'PUR':
                huc_regions = config['test_mode'].get('huc_regions', [])
                if not huc_regions:
                    log.warning("HUC regions not specified for PUR extent")
                
                holdout_basins = []
                for idx in holdout_indexs:
                    if idx < len(huc_regions):
                        holdout_basins.extend(huc_regions[idx])
                    else:
                        log.warning(f"Holdout index {idx} exceeds number of HUC regions")
                
                config['test_mode']['holdout_basins'] = holdout_basins
                
            elif extent == 'PUB':
                pub_ids = config['test_mode'].get('PUB_ids', [])
                if not pub_ids:
                    log.warning("PUB IDs not specified for PUB extent")
                
                holdout_basins = []
                for idx in holdout_indexs:
                    if idx < len(pub_ids):
                        holdout_basins.append(pub_ids[idx])
                    else:
                        log.warning(f"Holdout index {idx} exceeds number of PUB IDs")
                
                config['test_mode']['holdout_basins'] = holdout_basins

    if config['multimodel_type'] in ['none', 'None', '']:
        config['multimodel_type'] = None

    out_path = PathBuilder(config)
    config = out_path.write_path(config)
    
    config['dtype'] = eval(config['dtype'])

    return config


def save_model(config, model, model_name, epoch, create_dirs=False) -> None:
    """Save model state dict."""
    if create_dirs:
        out_path = PathBuilder(config)
        out_path.write_path(config)

    save_name = f"d{str(model_name)}_model_Ep{str(epoch)}.pt"

    full_path = os.path.join(config['out_path'], save_name)
    torch.save(model.state_dict(), full_path)


def save_outputs(config, preds_list, y_obs, create_dirs=False) -> None:
    """Save outputs from a model."""
    if create_dirs:
        out_path = PathBuilder(config)
        out_path.write_path(config)

    for key in preds_list[0].keys():
        if len(preds_list[0][key].shape) == 3:
            dim = 1
        else:
            dim = 0

        concatenated_tensor = torch.cat([d[key] for d in preds_list], dim=dim)
        file_name = key + ".npy"        

        np.save(os.path.join(config['testing_path'], file_name), concatenated_tensor.numpy())

    # Reading flow observation
    for var in config['train']['target']:
        item_obs = y_obs[:, :, config['train']['target'].index(var)]
        file_name = var + '_obs.npy'
        np.save(os.path.join(config['testing_path'], file_name), item_obs)


def print_config(config: Dict[str, Any]) -> None:
    """Print the current configuration settings.

    Parameters
    ----------
    config : dict
        Dictionary of configuration settings.
    """
    print()
    print("\033[1m" + "Current Configuration" + "\033[0m")
    print(f"  {'Experiment Mode:':<20}{config['mode']:<20}")
    if config['multimodel_type'] != None:
        print(f"  {'Ensemble Mode:':<20}{config['multimodel_type']:<20}")
    for i, mod in enumerate(config['dpl_model']['phy_model']['model']):
        print(f"  {f'Model {i+1}:':<20}{mod:<20}")
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f"  {'Data Source:':<20}{config['observations']['name']:<20}")
    if config['mode'] != 'test':
        print(f"  {'Train Range :':<20}{config['train']['start_time']:<20}{config['train']['end_time']:<20}")
    if config['mode'] != 'train':
        print(f"  {'Test Range :':<20}{config['test']['start_time']:<20}{config['test']['end_time']:<20}")
    if config['train']['start_epoch'] > 0:
        print(f"  {'Resume training from epoch:':<20}{config['train']['start_epoch']:<20}")
    print()
    if 'test_mode' in config:
        print("\033[1m" + "Test Mode Configuration" + "\033[0m")
        print(f"  {'Test Type:':<20}{config['test_mode'].get('type', 'Not specified'):<20}")
        
        if config['test_mode'].get('type') == 'spatial':
            print(f"  {'Spatial Extent:':<20}{config['test_mode'].get('extent', 'Not specified'):<20}")
            print(f"  {'Holdout Indices:':<20}{str(config['test_mode'].get('holdout_indexs', [])):<20}")
            
            if 'holdout_basins' in config['test_mode']:
                print(f"  {'Holdout Basins:':<20}{str(config['test_mode']['holdout_basins']):<60}")
                
            if config['test_mode'].get('extent') == 'PUR':
                print(f"  {'HUC Regions:':<20}{'Configured' if config['test_mode'].get('huc_regions') else 'Not configured':<20}")
            elif config['test_mode'].get('extent') == 'PUB':
                print(f"  {'PUB IDs:':<20}{'Configured' if config['test_mode'].get('PUB_ids') else 'Not configured':<20}")
                
        print()

    print("\033[1m" + "Model Parameters" + "\033[0m")
    print(f"  {'Train Epochs:':<20}{config['train']['epochs']:<20}{'Batch Size:':<20}{config['train']['batch_size']:<20}")
    print(f"  {'Dropout:':<20}{config['dpl_model']['nn_model']['dropout']:<20}{'Hidden Size:':<20}{config['dpl_model']['nn_model']['hidden_size']:<20}")
    print(f"  {'Warmup:':<20}{config['dpl_model']['phy_model']['warm_up']:<20}{'Concurrent Models:':<20}{config['dpl_model']['phy_model']['nmul']:<20}")
    print(f"  {'Loss Fn:':<20}{config['loss_function']['model']:<20}")
    print()

    if config['multimodel_type'] != None:
        print("\033[1m" + "Multimodel Parameters" + "\033[0m")
        print(f"  {'Mosaic:':<20}{config['multimodel']['mosaic']:<20}{'Dropout:':<20}{config['multimodel']['dropout']:<20}")
        print(f"  {'Learning Rate:':<20}{config['multimodel']['learning_rate']:<20}{'Hidden Size:':<20}{config['multimodel']['hidden_size']:<20}")
        print(f"  {'Scaling Fn:':<20}{config['multimodel']['scaling_function']:<20}{'Loss Fn:':<20}{config['multimodel']['loss_function']:<20}")
        print(f"  {'Range-bound Loss:':<20}{config['multimodel']['use_rb_loss']:<20}{'Loss Factor:':<20}{config['multimodel']['loss_factor']:<20}")
        print()

    print("\033[1m" + 'Machine' + "\033[0m")
    print(f"  {'Use Device:':<20}{str(config['device']):<20}")
    if config.get('use_multi_gpu', False):
        print(f"  {'Multi-GPU:':<20}{'Enabled':<20}{'Devices:':<20}{config.get('devices', 'Not specified'):<20}")
    print()

def find_shared_keys(*dicts: Dict[str, Any]) -> List[str]:
    """Find keys shared between multiple dictionaries.

    Parameters
    ----------
    *dicts : dict
        Variable number of dictionaries.

    Returns
    -------
    List[str]
        A list of keys shared between the input dictionaries.
    """
    if len(dicts) == 1:
        return list()

    # Start with the keys of the first dictionary.
    shared_keys = set(dicts[0].keys())

    # Intersect with the keys of all other dictionaries.
    for d in dicts[1:]:
        shared_keys.intersection_update(d.keys())

    return list(shared_keys)
