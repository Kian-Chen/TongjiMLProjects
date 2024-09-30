import torch
import torch.nn as nn
from layers.kan_layers import TaylorKANLayer, WaveKANLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.d_model = configs.d_model

        self.lstm_enc = nn.LSTM(input_size=self.channels, hidden_size=self.d_model, batch_first=True)
        self.lstm_dec = nn.LSTM(input_size=self.channels, hidden_size=self.d_model, batch_first=True)
        self.fc = TaylorKANLayer(self.d_model, self.channels, order=3, addbias=True)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
        enc_out, (hn, cn) = self.lstm_enc(x)
        dec_out, _ = self.lstm_dec(x_dec, (hn, cn))
        pred = self.fc(dec_out)
        return pred
