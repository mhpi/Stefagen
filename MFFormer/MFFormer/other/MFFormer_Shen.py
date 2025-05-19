import torch
from torch import nn
from MFFormer.layers.mask import MaskGenerator
from MFFormer.layers.positional_encoding import PositionalEncoding
from MFFormer.layers.transformer_layers import TransformerBackbone
from MFFormer.layers.features_embedding import TimeSeriesEncEmbedding, TimeSeriesDecEmbedding
# StaticEncEmbedding,  StaticDecEmbedding
from timm.models.vision_transformer import trunc_normal_


class StaticEncEmbedding(nn.Module):
    def __init__(self, feature_names, embed_dim, max_len=1000):
        super().__init__()
        self.feature_names = feature_names
        self.num_features = len(feature_names)
        self.embed_dim = embed_dim

        self.embeddings1 = nn.ModuleDict({name: nn.Linear(1, 64, bias=True) for name in feature_names})
        self.embeddings2 = nn.ModuleDict({name: nn.Linear(64, embed_dim, bias=True) for name in feature_names})

        self.masked_values = nn.Parameter(torch.randn(max_len, self.num_features), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        # truncated normal distribution
        trunc_normal_(self.masked_values, std=.02)

    def forward(self, x, feature_order, masked_index=None):
        """
        Input:
            x: batch_size, num_features
            feature_order: list of feature names
            masked_index: batch_size, num_features
            masked_vector: num_features, embed_dim
        """

        num_bs, num_features = x.shape

        if masked_index is not None:
            masked_vector = self.masked_values[:num_bs]
            masked_index = masked_index.to(masked_vector.device)
            x = torch.where(masked_index, masked_vector, x)

        x = x[:, None, :]
        x = x.permute(0, 2, 1)  # --> batch_size, num_features, 1

        # --> list: num_features * (batch_size, 64)
        embeds = [self.embeddings1[name](x[:, i:i + 1, :]) for i, name in enumerate(feature_order)]
        # --> list: num_features * (batch_size, embed_dim)
        embeds = [self.embeddings2[name](embeds[i]) for i, name in enumerate(feature_order)]
        # --> batch_size, num_features, embed_dim
        embeds = torch.concatenate(embeds, dim=1)

        return embeds


class StaticDecEmbedding(nn.Module):
    def __init__(self, feature_names, embed_dim):
        super().__init__()
        self.embeddings1 = nn.ModuleDict({name: nn.Linear(embed_dim, 64, bias=True) for name in feature_names})
        self.embeddings2 = nn.ModuleDict({name: nn.Linear(64, 1, bias=True) for name in feature_names})

    def forward(self, x, feature_order):
        """
        Input:
            x: batch_size, num_features, embed_dim
            feature_order: list of feature names
            batch_size, num_features
        """
        # --> list: num_features * (batch_size, 64)
        embeds = [self.embeddings1[name](x[:, i:i + 1, :]) for i, name in enumerate(feature_order)]
        # --> list: num_features * (batch_size, 1)
        embeds = [self.embeddings2[name](embeds[i]) for i, name in enumerate(feature_order)]
        # --> batch_size, num_features, 1
        embeds = torch.concatenate(embeds, dim=1)
        # --> batch_size, num_features
        embeds = embeds[:, :, 0]
        return embeds


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

        # norm the layers
        self.encoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # time series embedding
        self.time_series_embedding = TimeSeriesEncEmbedding(configs.time_series_variables, embed_dim)
        self.static_embedding = StaticEncEmbedding(configs.static_variables, embed_dim)

        # positional encoding
        self.positional_encoding = PositionalEncoding(embed_dim, dropout=configs.dropout)

        # define the mask function
        self.mask_generator = MaskGenerator(configs.mask_ratio_time_series, mask_ratio_static=configs.mask_ratio_static,
                                            min_window_size=configs.min_window_size,
                                            max_window_size=configs.max_window_size)

        # define the encoder
        self.encoder = TransformerBackbone(embed_dim, configs.num_enc_layers, d_ffd, configs.num_heads, configs.dropout)

        # define the decoder
        self.enc_2_dec_embedding = nn.Linear(embed_dim, embed_dim, bias=True)
        self.decoder = TransformerBackbone(embed_dim, configs.num_dec_layers, d_ffd, configs.num_heads, configs.dropout)

        # define the projection layer
        self.time_series_projection = TimeSeriesDecEmbedding(configs.time_series_variables, embed_dim)
        self.static_projection = StaticDecEmbedding(configs.static_variables, embed_dim)

        self.init_weights()

    def init_weights(self):
        # uniform_ distribution
        nn.init.uniform_(self.positional_encoding.position_embedding, -.02, .02)

    def forward(self, batch_data_dict, is_mask=True):

        batch_x = batch_data_dict['batch_x']  # [batch_size, seq_len, num_features]
        batch_c = batch_data_dict['batch_c']
        masked_time_series_index = batch_data_dict['batch_time_series_mask_index']
        masked_static_index = batch_data_dict['batch_static_mask_index']

        batch_size, seq_len, _ = batch_x.shape

        # generate the mask
        if is_mask:
            if masked_time_series_index.numel() == 0:
                unmasked_time_series_index, masked_time_series_index = self.mask_generator(batch_x.shape,
                                                                                           method="consecutive")
                unmasked_static_index, masked_static_index = self.mask_generator(batch_c.shape, method="isolated_point")
        else:
            unmasked_time_series_index, masked_time_series_index = None, None
            unmasked_static_index, masked_static_index = None, None

        # # replace the masked value with 0
        # batch_x = batch_x.masked_fill(masked_time_series_index, 0)
        # batch_c = batch_c.masked_fill(masked_static_index, 0)

        # features embedding, [batch_size, seq_len, d_model]
        enc_x = self.time_series_embedding(batch_x, feature_order=self.time_series_variables,
                                           masked_index=masked_time_series_index)

        # [batch_size, num_features, d_model]
        enc_c = self.static_embedding(batch_c, feature_order=self.static_variables, masked_index=masked_static_index)

        # concat the time series and static features
        enc_x = torch.cat([enc_x, enc_c], dim=1)  # [batch_size, seq_len+num_features, d_model]

        # add the positional encoding
        enc_x = self.positional_encoding(enc_x)

        # encoder
        hidden_states = self.encoder(enc_x)
        hidden_states = self.encoder_norm(hidden_states)

        hidden_states = self.enc_2_dec_embedding(hidden_states)

        # decoder
        dec_x = self.decoder(hidden_states)
        dec_x = self.decoder_norm(dec_x)

        dec_x_time_series = dec_x[:, :seq_len, :]  # [batch_size, seq_len, d_model]
        dec_x_static = dec_x[:, seq_len:, :]  # [batch_size, num_features d_model]

        # restore
        outputs_time_series = self.time_series_projection(dec_x_time_series, feature_order=self.time_series_variables)
        outputs_static = self.static_projection(dec_x_static, feature_order=self.static_variables)

        output_dict = {
            'outputs_time_series': outputs_time_series,
            'outputs_static': outputs_static,
            'masked_time_series_index': masked_time_series_index.to(outputs_time_series.device),
            'masked_static_index': masked_static_index.to(outputs_time_series.device),
            # 'unmasked_time_series_index': unmasked_time_series_index.to(outputs_time_series.device),
            # 'unmasked_static_index': unmasked_static_index.to(outputs_time_series.device),

        }

        return output_dict
