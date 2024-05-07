import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class TrackletSplitterModel(nn.Module):
#     def __init__(self, feature_size, num_layers, nhead, dropout=0.1):
#         super(TrackletSplitterModel, self).__init__()
#         self.model_type = 'Transformer'
#         self.pos_encoder = PositionalEncoding(feature_size, dropout)
#         encoder_layers = TransformerEncoderLayer(feature_size, nhead, feature_size * 4, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
#         self.fc_out = nn.Linear(feature_size, 2)  # Outputs 
#         self.init_weights()

#     def init_weights(self):
#         initrange = 0.1
#         self.fc_out.bias.data.zero_()
#         self.fc_out.weight.data.uniform_(-initrange, initrange)

#     def forward(self, src, src_mask):
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src, src_mask)
#         output = self.fc_out(output)
#         return torch.sigmoid(output)  # Returns a tensor of shape (seq_length, 2) for each frame

class TrackletSplitterModel(nn.Module):
    def __init__(self, feature_size, num_layers, nhead, seq_length, dropout=0.1):
        super(TrackletSplitterModel, self).__init__()
        self.seq_length = seq_length
        self.pos_encoder = PositionalEncoding(feature_size, dropout)
        encoder_layers = TransformerEncoderLayer(feature_size, nhead, feature_size * 4, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.fc_impurity = nn.Linear(feature_size, 1)  # Output for tracklet impurity
        self.fc_cutoff = nn.Linear(feature_size, seq_length)  # Output for cutoff frame prediction
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.fc_impurity.bias.data.zero_()
        self.fc_impurity.weight.data.uniform_(-initrange, initrange)
        self.fc_cutoff.bias.data.zero_()
        self.fc_cutoff.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        impurity_prob = torch.sigmoid(self.fc_impurity(output[:, -1, :]))  # Use the last timestep for impurity prediction
        cutoff_frame = self.fc_cutoff(output)
        return impurity_prob, cutoff_frame

class PositionalEncoding(nn.Module):
    def __init__(self, feature_size, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, feature_size, 2) * -(math.log(10000.0) / feature_size))
        pe = torch.zeros(max_len, 1, feature_size)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask