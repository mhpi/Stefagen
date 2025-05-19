"""Utilities for handling configuration objects safely."""

import logging
from typing import Any, Dict, Union
from omegaconf import OmegaConf, DictConfig

log = logging.getLogger(__name__)

def sanitize_config(config: Union[Dict, DictConfig]) -> Dict:
    """
    Recursively sanitize a configuration object to remove any values that
    can't be serialized by OmegaConf.
    
    Parameters
    ----------
    config : Dict or DictConfig
        Configuration object to sanitize
    
    Returns
    -------
    Dict
        A sanitized version of the configuration
    """
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    
    if not isinstance(config, dict):
        return config
    
    result = {}
    for key, value in config.items():
        # Skip problematic keys and types
        if key == 'dtype' or isinstance(value, type):
            continue
            
        # Recursively sanitize nested dictionaries
        if isinstance(value, dict):
            result[key] = sanitize_config(value)
        # Handle lists - they might contain dicts
        elif isinstance(value, list):
            result[key] = [
                sanitize_config(item) if isinstance(item, dict) else item
                for item in value
            ]
        # Simple values - keep as is
        else:
            result[key] = value
            
    return result

def safe_save_config(config: Union[Dict, DictConfig], filepath: str) -> bool:
    """
    Safely save a configuration to a YAML file, handling any serialization issues.
    
    Parameters
    ----------
    config : Dict or DictConfig
        Configuration to save
    filepath : str
        Path to save the configuration to
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    try:
        # Sanitize the configuration
        safe_config = sanitize_config(config)
        
        # Convert to OmegaConf and save
        conf_obj = OmegaConf.create(safe_config)
        with open(filepath, 'w') as f:
            f.write(OmegaConf.to_yaml(conf_obj))
        return True
    except Exception as e:
        log.error(f"Error saving configuration: {str(e)}")
        # Write error information to file
        try:
            with open(f"{filepath}.error", 'w') as f:
                f.write(f"Error saving configuration: {str(e)}")
        except Exception:
            pass
        return False