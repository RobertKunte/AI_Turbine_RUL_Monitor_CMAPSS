try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for LSTMRULPredictor. Please install torch."
    ) from exc

# We determine the input and output sizes
# X_final.shape was (N, SEQUENCE_LENGTH, NUMBER_OF_FEATURES)
# INPUT_SIZE = X_final.shape[2] # Number of sensors (features)

class LSTMRULPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRULPredictor, self).__init__()
        
        # 1. The LSTM layer: Processes the sequence
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True # Important: Batch dimension is first (Batch, Sequence, Features)
        )
        
        # 2. The final layer (Fully Connected): 
        # Reduces the LSTM output to our desired RUL value (Output_Size=1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 'h_n' and 'c_n' are the Hidden and Cell states (not directly needed here)
        # 'output' is the output of the LSTM layer for EACH time step
        output, (h_n, c_n) = self.lstm(x)
        
        # We are only interested in the output of the LAST time step of the sequence
        # as this represents the accumulated state of the entire history.
        last_output = output[:, -1, :] 
        
        # Pass the last output through the Fully Connected layer to get the RUL
        prediction = self.fc(last_output)
        return prediction

class LSTMRULWithAttention(nn.Module):
    """
    LSTM model with simple attention over time.
    Uses the same hyperparameter interface as the baseline model.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRULWithAttention, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2  # optional, can be set to 0.0
        )
        
        # Attention: score per time step
        self.attn = nn.Linear(hidden_size, 1)
        
        # Final regression layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (B, T, F)
        lstm_out, _ = self.lstm(x)          # (B, T, H)
        
        # Attention weights over time steps
        attn_scores  = self.attn(lstm_out)  # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        
        # Weighted sum over time dimension
        context = (lstm_out * attn_weights).sum(dim=1)    # (B, H)
        
        out = self.fc(context)                            # (B, 1)
        return out
