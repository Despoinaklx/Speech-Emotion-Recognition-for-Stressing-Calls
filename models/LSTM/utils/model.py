import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """(Bi)LSTM over padded sequences; classifies from the last hidden state."""

    def __init__(self, input_dim, num_classes, hidden_dim=256, num_layers=2, dropout=0., bidirectional=False):
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        self.dropout = nn.Dropout(dropout)

        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, num_classes)

    def forward(self, x, lengths):
        lengths = torch.clamp(lengths, min=1, max=x.size(1))

        # Packing makes the LSTM ignore padding
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)

        if self.lstm.bidirectional:
            # Concatenate last forward and backward states
            h_last = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden_dim*2]
        else:
            h_last = h_n[-1]  # [batch, hidden_dim]

        out = self.dropout(h_last)
        logits = self.fc(out)

        return logits
