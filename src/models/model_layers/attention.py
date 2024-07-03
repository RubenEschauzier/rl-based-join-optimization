# Written by: ast0414 (https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py)
import torch
import torch.nn as nn

from src.utils.model_utils.utils import masked_log_softmax


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

        self.all_weights = [self.W1, self.W2, self.vt]

    def forward(self, decoder_state, encoder_outputs, mask):

        # (batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)

        # (batch_size, 1 (unsqueezed), hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        # 1st line of Eq.(3) in the paper
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        u_i = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)

        # softmax with only valid inputs, excluding zero padded parts
        # log-softmax for a better numerical stability
        log_score = masked_log_softmax(u_i, mask, dim=-1)

        return log_score

    def devices(self):
        devices = []
        for weight in self.all_weights:
            for param in weight.parameters():
                devices.append(param.device)
        return devices
