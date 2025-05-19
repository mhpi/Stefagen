"""
3 basins:
Use GPU: cuda:0
>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>
Epoch: 1 loss: 0.358 cost time: 10.373
results saved in:  /data/jql6620/local/30.MFFormer/MFFormer/output/pretrain_time_series_MFFormer_time_series_CAMELS_Frederik_seq_len365_pred_len365_d_model256_n_heads8_e_layers2_d_layers1_d_ff512_Test/results
>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>
PRCP_nldas_extended. NSE: 0.810, KGE: 0.851, Corr: 0.902
SRAD_nldas_extended. NSE: 0.860, KGE: 0.909, Corr: 0.931
Tmax_nldas_extended. NSE: 0.881, KGE: 0.817, Corr: 0.966
Tmin_nldas_extended. NSE: 0.802, KGE: 0.808, Corr: 0.933
Vp_nldas_extended. NSE: 0.941, KGE: 0.940, Corr: 0.972
obs_mean: 238.143, obs_median: 15.6899852
pred_mean: 240.852, pred_median: 16.5807966

Use GPU: cuda:0
>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>
Epoch: 1 loss: 0.360 cost time: 9.197
results saved in:  /data/jql6620/local/30.MFFormer/MFFormer/output/pretrain_time_series_MFFormer_time_series_CAMELS_Frederik_seq_len365_pred_len365_d_model256_n_heads8_e_layers2_d_layers1_d_ff512_Test/results
>>>>>>>start testing >>>>>>>>>>>>>>>>>>>>>>>>>>
PRCP_nldas_extended. NSE: 0.810, KGE: 0.845, Corr: 0.900
SRAD_nldas_extended. NSE: 0.866, KGE: 0.894, Corr: 0.935
Tmax_nldas_extended. NSE: 0.879, KGE: 0.815, Corr: 0.966
Tmin_nldas_extended. NSE: 0.783, KGE: 0.825, Corr: 0.937
Vp_nldas_extended. NSE: 0.934, KGE: 0.943, Corr: 0.970
obs_mean: 238.143, obs_median: 15.6899852
pred_mean: 238.531, pred_median: 16.4148615
"""
import torch
from torch import nn
from MFFormer.layers.mask import MaskGenerator
from MFFormer.layers.positional_encoding import PositionalEncoding
from MFFormer.layers.transformer_layers import TransformerBackbone
from MFFormer.layers.features_embedding import TimeSeriesEncEmbedding, TimeSeriesDecEmbedding


# from MFFormer.utils.validation_tools import Validator


# define the Multiple Features for Time Series Transformer
class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()

        mlp_ratio = 4
        embed_dim = configs.d_model
        d_ffd = embed_dim * mlp_ratio

        self.embed_dim = embed_dim
        self.time_series_variables = configs.time_series_variables

        # norm the layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # time series embedding
        self.time_series_embedding = TimeSeriesEncEmbedding(configs.time_series_variables, embed_dim)

        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=configs.dropout)

        # define the mask function
        self.mask_generator = MaskGenerator(configs.mask_ratio_time_series, min_window_size=configs.min_window_size,
                                            max_window_size=configs.max_window_size)

        # define the encoder
        self.encoder = TransformerBackbone(embed_dim, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout)

        # define the decoder
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder = TransformerBackbone(embed_dim, configs.num_dec_layers, d_ffd, configs.num_heads, configs.dropout)

        # define the projection layer
        self.time_series_projection = TimeSeriesDecEmbedding(configs.time_series_variables, embed_dim)

        self.init_weights()

    def init_weights(self):
        # uniform_ distribution
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)

    def forward(self, batch_data_dict, is_mask=True):

        batch_x = batch_data_dict['batch_x']
        # batch_c = batch_data_dict['batch_c']
        batch_time_series_mask_index = batch_data_dict['batch_time_series_mask_index']

        # if batch_c is not None:
        #     input_data = Validator.combine_timeseries_and_statics(batch_x, batch_c) # (batch_size, seq_len, num_x + num_c)

        input_data = batch_x

        # input_data: [batch_size, seq_len, num_features]

        # generate the mask
        if is_mask:
            if batch_time_series_mask_index.numel() == 0:
                unmaksed_time_series_index, masked_time_series_index = self.mask_generator(input_data.shape)
            else:
                masked_time_series_index = batch_time_series_mask_index
                unmaksed_time_series_index = torch.logical_not(masked_time_series_index)
        else:
            unmaksed_time_series_index, masked_time_series_index = None, None

        # features embedding, [batch_size, seq_len, d_model]
        enc_x = self.time_series_embedding(input_data, feature_order=self.time_series_variables, masked_index=masked_time_series_index)

        # add the positional encoding
        enc_x = self.positional_encoding(enc_x)

        # encoder
        hidden_states = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)

        # decoder
        dec_x = self.decoder(hidden_states)
        dec_x = self.decoder_norm(dec_x)

        # restore
        outputs_time_series = self.time_series_projection(dec_x, feature_order=self.time_series_variables)

        output_dict = {
            'outputs_time_series': outputs_time_series,
            'unmaksed_time_series_index': unmaksed_time_series_index,
            'masked_time_series_index': masked_time_series_index
        }

        return output_dict
