import torch
import torch.nn as nn
import torch.nn.functional as F

class PointerNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=8):
        super(PointerNetwork, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.q_value = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim) or (batch_size, 1, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        encoder_outputs, (hidden, cell) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1)
        decoder_outputs, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
        context, attn_weights = self.attention(encoder_outputs, hidden[-1])
        q_values = self.q_value(context)
        
        return q_values, attn_weights

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, encoder_outputs, decoder_hidden):
        # Calculate attention scores
        attn_scores = torch.bmm(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights
