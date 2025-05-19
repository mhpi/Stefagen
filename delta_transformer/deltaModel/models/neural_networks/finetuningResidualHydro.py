import os
import re
import torch
import torch.nn as nn
import logging
import math
import warnings
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Union, Optional
from models.neural_networks.transformer_layers import TransformerBackbone
from models.neural_networks.positional_encoding import PositionalEncoding
from models.neural_networks.features_embedding import (
    TimeSeriesEncEmbedding, 
    StaticEncEmbedding
)
from models.neural_networks.MFFormer import Model as MFFormer
from torch.nn import Parameter
import torch.nn.functional as F

# Import CudnnLstm from lstm_models.py instead of defining it locally
from models.neural_networks.lstm_models import CudnnLstm

# Import needed for createMask
from models.neural_networks.dropout import DropMask, createMask

# Suppress warning for weights not part of contiguous chunk
warnings.filterwarnings("ignore", message=".*weights are not part of single contiguous chunk.*")

log = logging.getLogger(__name__)

# CudnnLstm class is now imported from models.neural_networks.lstm_models

class DualResidualAdapter(nn.Module):
    """Dual residual adapter with both static and time series residual paths."""
    def __init__(self, d_model: int, n_time_features: int, n_static_features: int):
        super().__init__()
        self.d_model = d_model
        self.n_time_features = n_time_features
        self.n_static_features = n_static_features
        
        # Time series specific transformation
        self.time_transform = nn.Sequential(
            nn.Linear(n_time_features, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Static features specific transformation
        self.static_transform = nn.Sequential(
            nn.Linear(n_static_features, d_model),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Combined transformation
        self.combined_transform = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Layer normalization for stability
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
    
    def forward(self, hidden_states, time_features, static_features):
        batch_size, seq_len, _ = hidden_states.shape
        
        # Transform time series features
        time_proj = self.time_transform(time_features)
        
        # Transform static features and expand to sequence length
        static_proj = self.static_transform(static_features)
        static_expanded = static_proj.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine all features
        combined = torch.cat([hidden_states, time_proj, static_expanded], dim=-1)
        combined_proj = self.combined_transform(combined)
        
        # Create dual residual paths
        output = self.layer_norm1(hidden_states + combined_proj)
        output = self.layer_norm2(output + time_proj + static_expanded)
        
        return output


class FineTunerResidualHydro(nn.Module):
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
            num_heads = 4
            dropout = nn_config.get('dropout', 0.1)
            
            self.model_config = {
                # Model architecture
                'd_model': d_model,
                'num_heads': num_heads,
                'dropout': dropout,
                
                # Settings from nn_model config
                'num_enc_layers': nn_config.get('num_enc_layers', 4),
                'num_dec_layers': nn_config.get('num_dec_layers', 2),
                'd_ffd': nn_config.get('d_ffd', 512),
                
                # Data configuration
                'time_series_variables': nn_config.get('forcings', []),
                'static_variables': nn_config.get('attributes', []),
                'pred_len': dpl_config.get('rho', 365),
                'pretrained_model': dpl_config.get('pretrained_model')
            }
            
            # For target variables, use forcings if target not specified
            self.model_config['target_variables'] = ny
            
            # Validate configuration
            missing_configs = []
            if not self.model_config['time_series_variables']:
                missing_configs.append("time_series_variables (forcings)")
            if not self.model_config['target_variables']:
                missing_configs.append("target_variables")

            if missing_configs:
                raise ValueError(f"Missing or invalid configuration: {', '.join(missing_configs)}")

        except Exception as e:
            log.error(f"Error processing config: {str(e)}")
            raise
            
        mfformer_config = type('Config', (), {
                    'd_model': self.model_config['d_model'],
                    'num_heads': self.model_config['num_heads'],
                    'dropout': self.model_config['dropout'],
                    'num_enc_layers': self.model_config['num_enc_layers'],
                    'num_dec_layers': self.model_config['num_dec_layers'],
                    'd_ffd': self.model_config['d_ffd'],
                    'time_series_variables': self.model_config['time_series_variables'],
                    'static_variables': self.model_config['static_variables'],
                    'static_variables_category': [],
                    'static_variables_category_dict': {},
                    'mask_ratio_time_series': nn_config.get('mask_ratio_time_series', 0.5),
                    'mask_ratio_static': nn_config.get('mask_ratio_static', 0.5),
                    'min_window_size': nn_config.get('min_window_size', 12),
                    'max_window_size': nn_config.get('max_window_size', 36),
                    'init_weight': config_dict.get('init_weight', 0.02),
                    'init_bias': config_dict.get('init_bias', 0.02),
                    'warmup_train': False,
                    'add_input_noise': False
                })

        # Initialize components
        built_model = MFFormer(mfformer_config).float()
        self.pretrained_model = self.load_pre_trained_model(built_model)
        
        # Get dimensions
        n_time_features = len(self.model_config['time_series_variables'])
        n_static_features = len(self.model_config['static_variables'])
        
        # Initialize dual residual adapter
        self.adapter = DualResidualAdapter(
            d_model=self.model_config['d_model'],
            n_time_features=n_time_features,
            n_static_features=n_static_features
        )
        
        # Prepare LSTM input by combining features
        self.lstm_input_dim = self.model_config['d_model'] + n_time_features + n_static_features
        self.pre_lstm = nn.Linear(self.lstm_input_dim, self.model_config['d_model'])
        
        # Initialize decoder LSTM
        # Note: Use inputSize and hiddenSize parameters instead of nx and hidden_size
        self.decoder = CudnnLstm(
            inputSize=self.model_config['d_model'], 
            hiddenSize=self.model_config['d_model'],
            dr=self.model_config['dropout']
        )
        
        # Post-processing with residual connections
        self.post_lstm = nn.Linear(
            self.model_config['d_model'] + n_time_features + n_static_features,
            self.model_config['d_model']
        )
        
        # Final projection layer
        self.projection = nn.Linear(
            self.model_config['d_model'],
            self.model_config['target_variables']
        )
        
        log.info("Successfully initialized FineTunerDualResidualHydro model")

    def load_pre_trained_model(self, model: nn.Module) -> nn.Module:
        """Load pretrained model weights with updated PyTorch 2.6+ compatibility."""
        pretrained_model = self.model_config['pretrained_model']
        if not pretrained_model:
            log.warning("No pretrained model path provided, using randomly initialized weights")
            return model

        try:
            if os.path.isdir(pretrained_model):
                checkpoint_list = [f for f in os.listdir(pretrained_model) if re.search(r'^.+_[\d]*.pt$', f) is not None]
                if not checkpoint_list:
                    raise ValueError(f"No checkpoint files found in {pretrained_model}")
                checkpoint_list.sort()
                checkpoint_file = os.path.join(pretrained_model, checkpoint_list[-1])
            elif os.path.isfile(pretrained_model):
                checkpoint_file = pretrained_model
            else:
                raise ValueError('pretrained_model is not a valid file or directory')

            # Add numpy scalar to safe globals and load with proper settings
            import numpy as np
            from torch.serialization import safe_globals, add_safe_globals
            
            # Add numpy scalar to safe globals
            add_safe_globals([np.core.multiarray.scalar])
            
            # Load checkpoint using context manager for safe globals
            with safe_globals([np.core.multiarray.scalar]):
                checkpoint_dict = torch.load(checkpoint_file, map_location='cpu', weights_only=False)

            pretrained_state_dict = checkpoint_dict['model_state_dict']
            current_state_dict = model.state_dict()
            
            # Copy matching parameters
            for name, param in pretrained_state_dict.items():
                if name in current_state_dict and current_state_dict[name].size() == param.size():
                    current_state_dict[name].copy_(param)

            # Handle distributed training case
            if list(current_state_dict.keys())[0].startswith('module.'):
                new_state_dict = {k.replace('module.', ''): v for k, v in current_state_dict.items()}
                model.load_state_dict(new_state_dict)
            else:
                model.load_state_dict(current_state_dict)

            # Freeze pretrained parameters
            for param in model.parameters():
                param.requires_grad = False
                
            return model
            
        except Exception as e:
            log.error(f"Error loading pretrained model: {str(e)}")
            raise

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass with dual residual connections for both static and time series data."""
        # Get input data
        xc_nn_norm = data_dict['xc_nn_norm']  # [batch, time, features]
        
        # Track shapes for debugging
        input_shape = xc_nn_norm.shape
        log.debug(f"Input shape: {input_shape}")
        
        # Split time series and static features
        n_time_series = len(self.model_config['time_series_variables'])
        batch_x = xc_nn_norm[..., :n_time_series]  # [batch, time, time_features]
        batch_c = xc_nn_norm[..., n_time_series:][..., 0, :]  # [batch, static_features]
        
        log.debug(f"Time series shape: {batch_x.shape}, Static shape: {batch_c.shape}")
        
        # Handle missing values
        x_mask = torch.isnan(batch_x)
        c_mask = torch.isnan(batch_c)
        
        x_median = torch.nanmedian(batch_x) if not torch.isnan(batch_x).all() else torch.tensor(0.0)
        c_median = torch.nanmedian(batch_c) if not torch.isnan(batch_c).all() else torch.tensor(0.0)
        
        batch_x = batch_x.masked_fill(x_mask, x_median)
        batch_c = batch_c.masked_fill(c_mask, c_median)
        
        # Process with transformer
        with torch.amp.autocast('cuda', enabled=False):
            # Get embeddings from pretrained model
            enc_x = self.pretrained_model.time_series_embedding(
                batch_x,
                feature_order=self.model_config['time_series_variables']
            )
            
            enc_c = self.pretrained_model.static_embedding(
                batch_c,
                feature_order=self.model_config['static_variables']
            )
            
            # Save original embeddings for residual connections
            orig_time_features = batch_x
            orig_static_features = batch_c
            
            # Process through transformer encoder
            enc_combined = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
            enc_combined = self.pretrained_model.positional_encoding(enc_combined)
            
            hidden_states = self.pretrained_model.encoder(enc_combined)
            hidden_states = self.pretrained_model.encoder_norm(hidden_states)
            hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
            
            # Extract time series portion
            hidden_states = hidden_states[:, :batch_x.size(1), :]
            
            # Apply dual residual adapter
            adapted = self.adapter(hidden_states, orig_time_features, orig_static_features)
            
            log.debug(f"Adapted shape: {adapted.shape}")
            
            # Prepare LSTM input with residual connections
            # Expand static features to match sequence length
            static_expanded = orig_static_features.unsqueeze(1).expand(-1, adapted.size(1), -1)
            
            # Concatenate adapted features with original inputs for rich representation
            lstm_input_combined = torch.cat([adapted, orig_time_features, static_expanded], dim=-1)
            
            # Project to LSTM dimension
            lstm_input = self.pre_lstm(lstm_input_combined)
            
            # Permute for LSTM [seq, batch, features]
            lstm_input = lstm_input.permute(1, 0, 2)
            
            log.debug(f"LSTM input shape: {lstm_input.shape}")
            
            # Process through LSTM
            # Use doDropMC and dropoutFalse parameter names to match the imported CudnnLstm
            lstm_output, _ = self.decoder(
                lstm_input,
                doDropMC=False,
                dropoutFalse=(not self.training)
            )
            
            # Convert back to [batch, seq, features]
            lstm_output = lstm_output.permute(1, 0, 2)
            
            log.debug(f"LSTM output shape: {lstm_output.shape}")
            
            # Add final residual connections
            # Reintroduce time series and static information
            post_lstm_combined = torch.cat([lstm_output, orig_time_features, static_expanded], dim=-1)
            
            # Process through post-LSTM layer with residual connection
            post_output = self.post_lstm(post_lstm_combined)
            final_output = post_output + lstm_output  # Residual connection
            
            # Final projection
            outputs = self.projection(final_output)
            
            # Return as [time, batch, params]
            outputs = outputs.permute(1, 0, 2)
            
            log.debug(f"Final output shape: {outputs.shape}")
            
            return outputs