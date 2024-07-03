# Written by: ast0414 (https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py)
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(Encoder, self).__init__()

        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=self.batch_first,
                                                   enforce_sorted=False)
        # Forward pass through RNN
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # Return output and final hidden state
        return outputs, hidden

    def devices(self):
        return [param.device for param in self.rnn.parameters()]
