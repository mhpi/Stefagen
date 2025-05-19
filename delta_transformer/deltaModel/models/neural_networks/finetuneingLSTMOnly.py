import os
import re
import torch
import torch.nn as nn
import logging
import math
import warnings
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Union, Optional
from torch.nn import Parameter
import torch.nn.functional as F

# Import needed for CudnnLstm
from models.neural_networks.dropout import DropMask, createMask
from models.neural_networks.lstm_models import CudnnLstm

# Suppress warning for weights not part of contiguous chunk
warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")

log = logging.getLogger(__name__)



class LSTMOnlyFineTuner(nn.Module):
    def __init__(self, config: Union[Dict, DictConfig], ny) -> None:
        super().__init__()
        
        try:
            # Convert config if needed
            if isinstance(config, DictConfig):
                config_dict = OmegaConf.to_container(config, resolve=True)
            else:
                config_dict = config
            
            log.info(f"Processing config: {config_dict}")
            
            # Extract configuration - handle both full config and dpl_model config cases
            if 'nn_model' in config_dict:  # Already at dpl_model level
                dpl_config = config_dict
                nn_config = config_dict['nn_model']
            else:  # Full config structure
                dpl_config = config_dict.get('dpl_model', {})
                nn_config = dpl_config.get('nn_model', {})
            
            # Extract settings
            d_model = nn_config.get('hidden_size', 256)
            dropout = nn_config.get('dropout', 0.1)
            
            self.model_config = {
                # Model architecture
                'd_model': d_model,
                'dropout': dropout,
                
                # Data configuration
                'time_series_variables': nn_config.get('forcings', []),
                'static_variables': nn_config.get('attributes', []),
                'pred_len': dpl_config.get('rho', 365)
            }
            
            # For target variables, use ny parameter
            self.model_config['target_variables'] = ny
            
            # Validate configuration
            missing_configs = []
            if not self.model_config['time_series_variables']:
                missing_configs.append("time_series_variables (forcings)")
            if not self.model_config['target_variables']:
                missing_configs.append("target_variables")

            if missing_configs:
                raise ValueError(f"Missing configuration: {', '.join(missing_configs)}")

        except Exception as e:
            log.error(f"Error processing config: {str(e)}")
            raise
        
        # Input dimensions
        n_time_features = len(self.model_config['time_series_variables'])
        n_static_features = len(self.model_config['static_variables'])
        
        # Similar to the original but without pretrained model - use direct inputs
        # Prepare LSTM input by combining features
        self.lstm_input_dim = n_time_features + n_static_features
        self.pre_lstm = nn.Linear(self.lstm_input_dim, self.model_config['d_model'])
        
        # Initialize decoder LSTM (same as original)
        self.decoder = CudnnLstm(
            nx=self.model_config['d_model'],
            hidden_size=self.model_config['d_model'],
            dr=self.model_config['dropout']
        )
        
        # Post-processing (simplified but similar structure)
        self.post_lstm = nn.Linear(
            self.model_config['d_model'] + n_time_features + n_static_features,
            self.model_config['d_model']
        )
        
        # Final projection layer (same)
        self.projection = nn.Linear(
            self.model_config['d_model'],
            self.model_config['target_variables']
        )
        
        log.info("Successfully initialized LSTMOnlyFineTuner model")

  

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with direct input to LSTM, without transformer components."""
        # Get input data (same as original)
        xc_nn_norm = data_dict['xc_nn_norm']  # [batch, time, features]
        
        # Split time series and static features (same approach)
        n_time_series = len(self.model_config['time_series_variables'])
        batch_x = xc_nn_norm[..., :n_time_series]  # [batch, time, time_features]
        batch_c = xc_nn_norm[..., n_time_series:][..., 0, :]  # [batch, static_features]
        
        # Handle missing values (same approach)
        x_mask = torch.isnan(batch_x)
        c_mask = torch.isnan(batch_c)
        
        x_median = torch.nanmedian(batch_x) if not torch.isnan(batch_x).all() else torch.tensor(0.0)
        c_median = torch.nanmedian(batch_c) if not torch.isnan(batch_c).all() else torch.tensor(0.0)
        
        batch_x = batch_x.masked_fill(x_mask, x_median)
        batch_c = batch_c.masked_fill(c_mask, c_median)
        
        # Save original inputs for residual connections
        orig_time_features = batch_x
        orig_static_features = batch_c
        
        # Expand static features to match sequence length
        static_expanded = orig_static_features.unsqueeze(1).expand(-1, orig_time_features.size(1), -1)
        
        # Instead of transformer, directly combine inputs
        combined_input = torch.cat([orig_time_features, static_expanded], dim=-1)
        
        # Project to LSTM dimension
        lstm_input = self.pre_lstm(combined_input)
        
        # Permute for LSTM [seq, batch, features]
        lstm_input = lstm_input.permute(1, 0, 2)
        
        # Process through LSTM (same as original)
        lstm_output, _ = self.decoder(
            lstm_input,
            do_drop_mc=False,
            dr_false=(not self.training)
        )
        
        # Convert back to [batch, seq, features]
        lstm_output = lstm_output.permute(1, 0, 2)
        
        # Add final residual connections (similar to original but simplified)
        post_lstm_combined = torch.cat([lstm_output, orig_time_features, static_expanded], dim=-1)
        
        # Process through post-LSTM layer with residual connection
        post_output = self.post_lstm(post_lstm_combined)
        final_output = post_output + lstm_output  # Residual connection
        
        # Final projection
        outputs = self.projection(final_output)
        
        # Return as [time, batch, params]
        outputs = outputs.permute(1, 0, 2)
        
        return outputs