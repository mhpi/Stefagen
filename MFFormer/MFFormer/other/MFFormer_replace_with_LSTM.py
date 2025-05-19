import numpy as np
import torch
from torch import nn
from MFFormer.layers.mask import MaskGenerator
from MFFormer.layers.positional_encoding import PositionalEncoding
from MFFormer.layers.transformer_layers import TransformerBackbone
from MFFormer.layers.features_embedding import TimeSeriesEncEmbedding, TimeSeriesDecEmbedding, StaticEncEmbedding, \
    StaticDecEmbedding


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
        self.static_variables = configs.static_variables
        self.static_variables_category = configs.static_variables_category
        self.static_variables_category_num = [len(configs.static_variables_category_dict[x]['class_to_index']) for x in configs.static_variables_category]
        self.static_variables_numeric = [var for var in self.static_variables if var not in self.static_variables_category]

        # norm the layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # time series embedding
        self.time_series_embedding = TimeSeriesEncEmbedding(configs.time_series_variables, embed_dim)
        self.static_embedding = StaticEncEmbedding(self.static_variables_numeric, embed_dim,
                                                   categorical_features=self.static_variables_category,
                                                   categorical_features_num=self.static_variables_category_num)

        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=configs.dropout)

        # define the mask function
        self.mask_generator = MaskGenerator(configs.mask_ratio_time_series, mask_ratio_static=configs.mask_ratio_static,
                                            min_window_size=configs.min_window_size,
                                            max_window_size=configs.max_window_size)

        # define the encoder
        self.encoder = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

        # define the decoder
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim, batch_first=True)

        # define the projection layer
        self.time_series_projection = TimeSeriesDecEmbedding(configs.time_series_variables, embed_dim)
        self.static_projection = StaticDecEmbedding(self.static_variables_numeric, embed_dim,
                                                    categorical_features=self.static_variables_category,
                                                    categorical_features_num=self.static_variables_category_num)

        self.init_weights()

    def init_weights(self):
        # uniform_ distribution
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)

    def forward(self, batch_data_dict, is_mask=True):

        batch_x = batch_data_dict['batch_x']  # [batch_size, seq_len, num_features]
        batch_c = batch_data_dict['batch_c']
        masked_time_series_index = batch_data_dict['batch_time_series_mask_index'].to(batch_x.device)
        masked_static_index = batch_data_dict['batch_static_mask_index'].to(batch_x.device)

        # generate the mask
        if is_mask:
            if masked_time_series_index.numel() == 0:
                _, masked_time_series_index = self.mask_generator(batch_x.shape,
                                                                  method="consecutive")
                _, masked_static_index = self.mask_generator(batch_c.shape, method="isolated_point")

            # missing data mask
            masked_missing_time_series_index = torch.isnan(batch_x)
            masked_missing_static_index = torch.isnan(batch_c)
            # replace the missing value with 0
            batch_x = batch_x.masked_fill(masked_missing_time_series_index, 0)
            batch_c = batch_c.masked_fill(masked_missing_static_index, 0)

            # combine the mask
            masked_time_series_index = masked_time_series_index | masked_missing_time_series_index
            masked_static_index = masked_static_index | masked_missing_static_index

        else:
            unmasked_time_series_index, masked_time_series_index = None, None
            unmasked_static_index, masked_static_index = None, None

        # # replace the masked value with 0
        # batch_x = batch_x.masked_fill(masked_time_series_index, 0)
        # batch_c = batch_c.masked_fill(masked_static_index, 0)

        # features embedding, [batch_size, seq_len, d_model]
        enc_x = self.time_series_embedding(batch_x, feature_order=self.time_series_variables,
                                           masked_index=masked_time_series_index)
        enc_c = self.static_embedding(batch_c, feature_order=self.static_variables, masked_index=masked_static_index)

        # concat the time series and static features
        enc_x = torch.cat([enc_x, enc_c[:, None, :]], dim=1)

        # add the positional encoding
        enc_x = self.positional_encoding(enc_x)

        # encoder
        hidden_states, (h_n, c_n) = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)

        # decoder
        dec_x, (h_n, c_n) = self.decoder(hidden_states)
        dec_x = self.decoder_norm(dec_x)

        dec_x_time_series = dec_x[:, :-1, :]  # [batch_size, seq_len, d_model]
        dec_x_static = dec_x[:, -1, :]  # [batch_size, d_model]

        # restore
        outputs_time_series = self.time_series_projection(dec_x_time_series, feature_order=self.time_series_variables)
        outputs_static, static_variables_dec_index_start, static_variables_dec_index_end = self.static_projection(dec_x_static, feature_order=self.static_variables, mode=batch_data_dict['mode'])

        output_dict = {
            'outputs_time_series': outputs_time_series,
            'outputs_static': outputs_static,
            'masked_time_series_index': masked_time_series_index,
            'masked_static_index': masked_static_index,
            # 'unmasked_time_series_index': unmasked_time_series_index.to(outputs_time_series.device),
            # 'unmasked_static_index': unmasked_static_index.to(outputs_time_series.device),
            "masked_missing_time_series_index": masked_missing_time_series_index,
            "masked_missing_static_index": masked_missing_static_index,
            'static_variables_dec_index_start': static_variables_dec_index_start,
            'static_variables_dec_index_end': static_variables_dec_index_end

        }

        return output_dict
