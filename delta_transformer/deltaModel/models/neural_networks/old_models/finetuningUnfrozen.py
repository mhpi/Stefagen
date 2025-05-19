import os
import re
import torch
import torch.nn as nn
import logging
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any, Union
from models.neural_networks.transformer_layers import TransformerBackbone
from models.neural_networks.positional_encoding import PositionalEncoding
from models.neural_networks.features_embedding import (
    TimeSeriesEncEmbedding, 
    StaticEncEmbedding
)
from models.neural_networks.MFFormer import Model as MFFormer

log = logging.getLogger(__name__)

class FineTunerUnfrozen(nn.Module):
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
            
            # log.info(f"Using NN config: {nn_config}")
            
            # Extract additional settings from parent config or use defaults
            d_model = nn_config.get('hidden_size', 256)  # Use hidden_size as d_model
            num_heads = 4  # Default value
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
                'adapter_type': nn_config.get('adapter_type', 'feedforward'),
                
                # Data configuration
                'time_series_variables': nn_config.get('forcings', []),
                'static_variables': nn_config.get('attributes', []),
                'pred_len': dpl_config.get('rho', 365),
                'pretrained_model': config_dict.get('pretrained_model')
            }
            
            # For target variables, use forcings if target not specified
            self.model_config['target_variables'] = ny
            
            # log.info(f"Initialized model config: {self.model_config}")

            # Validate configuration
            missing_configs = []
            if not self.model_config['time_series_variables']:
                missing_configs.append("time_series_variables (forcings)")
            if not self.model_config['target_variables']:
                missing_configs.append("target_variables")
            if self.model_config['adapter_type'] not in ['feedforward', 'conv', 'gated']:
                missing_configs.append(f"valid adapter_type (got {self.model_config['adapter_type']})")

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
                    'static_variables_category': [],  # Empty list if not using categorical variables
                    'static_variables_category_dict': {},  # Empty dict if not using categorical variables
                    'mask_ratio_time_series': nn_config.get('mask_ratio_time_series', 0.5),
                    'mask_ratio_static': nn_config.get('mask_ratio_static', 0.5),
                    'min_window_size': nn_config.get('min_window_size', 12),
                    'max_window_size': nn_config.get('max_window_size', 36),
                    'init_weight': config_dict.get('init_weight', 0.02),
                    'init_bias': config_dict.get('init_bias', 0.02),
                    'warmup_train': False,  # Set to False for fine-tuning
                    'add_input_noise': False  # Set to False for fine-tuning
                })

        # Initialize components
        built_model = MFFormer(mfformer_config).float()
        self.pretrained_model = self.load_pre_trained_model(built_model)
        self.adapter = self._initialize_adapter()
        
        # Initialize decoder and projection layers
        self.decoder = nn.LSTM(
            input_size=self.model_config['d_model'], 
            hidden_size=self.model_config['d_model'],
            batch_first=True
        )
        self.projection = nn.Linear(
            self.model_config['d_model'],
            self.model_config['target_variables']
        )
        
        log.info("Successfully initialized FineTuner model")

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
            # for param in model.parameters():
            #     param.requires_grad = False
                
            return model
            
        except Exception as e:
            log.error(f"Error loading pretrained model: {str(e)}")
            raise
    
    def _initialize_adapter(self) -> nn.Module:
        """Initialize the adapter layer based on config type."""
        adapter_type = self.model_config['adapter_type']
        d_model = self.model_config['d_model']
        n_features = len(self.model_config['time_series_variables'])

        if adapter_type == 'feedforward':
            return nn.Sequential(
                nn.Linear(d_model + n_features, d_model * 2),
                nn.ReLU(),
                nn.Linear(d_model * 2, d_model)
            )
        elif adapter_type == 'conv':
            return nn.Sequential(
                nn.Conv1d(d_model + n_features, d_model * 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(d_model * 2, d_model, kernel_size=3, padding=1)
            )
        elif adapter_type == 'gated':
            return GatedAdapter(d_model, n_features)
        else:
            return nn.Identity()

    def forward(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced NaN handling and gradient checks."""
        # Input processing with dimension checks
        xc_nn_norm = data_dict['xc_nn_norm']  # [batch, time, features]
        # xc_nn_norm = self.debug_tensor(xc_nn_norm, "input")
        
        # Split and validate dimensions
        n_time_series = len(self.model_config['time_series_variables'])
        batch_x = xc_nn_norm[..., :n_time_series]
        batch_c = xc_nn_norm[..., n_time_series:][..., 0, :]
        
        # log.info(f"Input shapes - batch_x: {batch_x.shape}, batch_c: {batch_c.shape}")

        # Handle missing values with robust statistics
        x_mask = torch.isnan(batch_x)
        c_mask = torch.isnan(batch_c)
        
        x_median = torch.nanmedian(batch_x) if not torch.isnan(batch_x).all() else torch.tensor(0.0)
        c_median = torch.nanmedian(batch_c) if not torch.isnan(batch_c).all() else torch.tensor(0.0)
        
        batch_x = batch_x.masked_fill(x_mask, x_median)
        batch_c = batch_c.masked_fill(c_mask, c_median)
        
        # batch_x = self.debug_tensor(batch_x, "batch_x")
        # batch_c = self.debug_tensor(batch_c, "batch_c")

        # Use gradient clipping and scaling for numerical stability
        max_norm = 1.0
        with torch.cuda.amp.autocast(enabled=False):
            # Embedding
            enc_x = self.pretrained_model.time_series_embedding(
                batch_x,
                feature_order=self.model_config['time_series_variables']
            )
            # enc_x = self.debug_tensor(enc_x, "enc_x")
            
            enc_c = self.pretrained_model.static_embedding(
                batch_c,
                feature_order=self.model_config['static_variables']
            )
            # enc_c = self.debug_tensor(enc_c, "enc_c")

            # Combine embeddings with scaled addition
            enc_x = torch.cat([enc_x, enc_c[:, None, :]], dim=1)
            enc_x = self.pretrained_model.positional_encoding(enc_x)
            # enc_x = self.debug_tensor(enc_x, "combined_enc")

            # Apply encoder with gradient clipping
            if hasattr(self.pretrained_model.encoder, 'clip_grad_norm_'):
                torch.nn.utils.clip_grad_norm_(self.pretrained_model.encoder.parameters(), max_norm)
            
            hidden_states = self.pretrained_model.encoder(enc_x)
            # hidden_states = self.debug_tensor(hidden_states, "encoder_output")
            
            # Apply layer normalization for stability
            hidden_states = self.pretrained_model.encoder_norm(hidden_states)
            hidden_states = self.pretrained_model.enc_2_dec_embedding(hidden_states)
            hidden_states = hidden_states[:, :batch_x.size(1), :]
            # hidden_states = self.debug_tensor(hidden_states, "normalized_hidden")

            # Adapter application with type checking
            if isinstance(self.adapter, nn.Identity):
                adapted = hidden_states
            else:
                if isinstance(self.adapter, nn.Sequential) and isinstance(self.adapter[0], nn.Conv1d):
                    adapter_input = torch.cat([hidden_states, batch_x], dim=-1).transpose(1, 2)
                    adapted = self.adapter(adapter_input).transpose(1, 2)
                else:
                    adapted = self.adapter(hidden_states, batch_x)
            
            # adapted = self.debug_tensor(adapted, "adapted")

            # Decode with gradient clipping
            dec_output, _ = self.decoder(adapted)
            # dec_output = self.debug_tensor(dec_output, "decoder_output")

            # Final projection with scaled outputs
            outputs = self.projection(dec_output)
            # outputs = self.debug_tensor(outputs, "final_outputs")

            # Ensure output shape consistency
            if len(outputs.shape) == 3:
                outputs = outputs.permute(1, 0, 2)  # [time, batch, params]
                
            # log.info(f"Final output shape: {outputs.shape}")
            return outputs

   

class GatedAdapter(nn.Module):
    """Gated adapter layer for fine-tuning."""
    def __init__(self, hidden_size: int, input_size: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size + input_size, hidden_size)
        self.transform = nn.Linear(hidden_size + input_size, hidden_size)

    def forward(self, h: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([h, x], dim=-1)
        gate = torch.sigmoid(self.gate(combined))
        transformed = self.transform(combined)
        return h + gate * transformed