import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadedDotAttention(nn.Module):
    def __init__(self, num_heads, features_size, dropout=0.1):
        super(MultiHeadedDotAttention, self).__init__()
        self.d_model = features_size
        self.num_heads = num_heads
        self.d_k = self.d_model // self.num_heads

        # Create linear projections
        self.query_linear = nn.Linear(features_size, features_size)
        self.key_linear = nn.Linear(features_size, features_size)
        self.value_linear = nn.Linear(features_size, features_size)

        self.aoa_layer = nn.Sequential(
            nn.Linear(features_size * 2, features_size * 2),
            nn.GLU()
        )
        self.output_linear = nn.Linear(features_size, features_size)

        self.dropout = nn.Dropout(dropout)

    def attention(self, query, key, value, dropout, att_mask = None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)

        if att_mask is not None:
          scores = scores.masked_fill(att_mask[:, None, None, :] == 1, float('-inf'))
        p_attn = F.softmax(scores, dim=-1)
        p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value)

    def forward(self, query, key, value, use_aoa = False, att_mask = None):
        batch_size = query.size(0)

        query_ = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key_ = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value_ = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attended = self.attention(query_, key_, value_, self.dropout, att_mask)

        # Concat using view
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, -1, self.d_model)

        # Attention on Attention
        if use_aoa:
          aoa_output = self.aoa_layer(torch.cat([attended, query], dim = 2))

        return self.output_linear(aoa_output)

class ResidualConnection(nn.Module):
    def __init__(self, _size, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(_size)

    def forward(self, x, att_features):
        return x + self.dropout(self.norm(att_features))

class AoA_Refiner_Layer(nn.Module):
    def __init__(self, features_size, num_heads, dropout=0.1):
        super(AoA_Refiner_Layer, self).__init__()
        self.attn = MultiHeadedDotAttention(num_heads, features_size)
        self.res_connection = ResidualConnection(features_size)

    def forward(self, x):
        att_features = self.attn(x, x, x, use_aoa = True)
        refined_features = self.res_connection(x, att_features)

        return refined_features


class AoA_Refiner_Core(nn.Module):
    def __init__(self, num_heads, stack_layers, features_size):
        super(AoA_Refiner_Core, self).__init__()

        self.layers = nn.ModuleList([AoA_Refiner_Layer(features_size, num_heads) for _ in range(stack_layers)])
        # self.layer = AoA_Refiner_Layer(features_size, num_heads)

        self.norm = nn.LayerNorm(features_size)

    def forward(self, x):

        for layer in self.layers:
            x = layer(x)
        # x = self.layer(x)

        return self.norm(x)


class AoA_Decoder_Core(nn.Module):
  def __init__(self, embedding_layer, num_heads, features_size, embedding_size, vocab_size):
    super(AoA_Decoder_Core, self).__init__()

    self.out_dropout = nn.Dropout(0.1)
    self.norm = nn.LayerNorm(embedding_size*2)

    self.resize_features = nn.Linear(features_size, embedding_size)

    self.embedding_layer = embedding_layer
    self.att_lstm = nn.LSTM(embedding_size * 2, embedding_size, num_layers = 2)
    self.multi_head = MultiHeadedDotAttention(num_heads, embedding_size)

    self.aoa_layer = nn.Sequential(
        nn.Linear(features_size * 2, features_size * 2),
        nn.GLU()
    )

    self.residual_connect = ResidualConnection(features_size)
    self.out_linear = nn.Linear(features_size, vocab_size)

  def forward(self, features, captions_ids, captions_mask = None):
    batch_size = features.size(0)
    sequence_length = captions_ids.size(1)

    # Prepare Img Features
    features = self.resize_features(features) # batch_size, img_size, embedding_size
    features_ = torch.mean(features, dim = 1).unsqueeze(dim=1).expand(batch_size, sequence_length, features_size) # batch_size, sequence_length, embedding_size

    # Embedding Captions
    embedded_captions = self.embedding_layer(captions_ids) # batch_size, sequence_length, embedding_size

    # Prepare Inputs
    input_concat = self.norm(torch.cat([features_, embedded_captions], dim = 2)) # batch_size, sequence_length, embedding_size * 2

    # LSTM
    output, (h_att, c_att) = self.att_lstm(input_concat) # batch_size, sequence_length, embedding_size

    # Calculate Attention
    att = self.multi_head(output, features, features, use_aoa = False) # batch_size, sequence_length, embedding_size

    # Applying AoA
    ctx_input = torch.cat([att, output], dim = 2) # batch_size, sequence_length, embedding_size * 2
    output_ = self.aoa_layer(ctx_input) # batch_size, sequence_length, embedding_size

    # Add Residual Connect
    residual_aoa = self.residual_connect(output_, output) # batch_size, sequence_length, embedding_size

    # Output
    return self.out_linear(self.out_dropout(residual_aoa)) # batch_size, sequence_length, vocab_size


class AoA_Model(nn.Module):
  def __init__(self, embedding_layer, num_heads, stack_layers, features_size, embedding_size, vocab_size):
    super(AoA_Model, self).__init__()

    self.refiner_layer = AoA_Refiner_Core(num_heads, stack_layers, features_size)
    self.decoder_layer = AoA_Decoder_Core(embedding_layer, num_heads, features_size, embedding_size, vocab_size)

    self.initialize_weights()

  def initialize_weights(self):
      for m in self.modules():
          if hasattr(m, 'weight') and m.weight.dim() > 1:
              nn.init.xavier_uniform_(m.weight.data)

  def forward(self, features, captions_ids, captions_mask):
    refined_features = self.refiner_layer(features) # batch_size, img_size, features_size
    decoded_outputs = self.decoder_layer(refined_features, captions_ids, captions_mask)

    return decoded_outputs