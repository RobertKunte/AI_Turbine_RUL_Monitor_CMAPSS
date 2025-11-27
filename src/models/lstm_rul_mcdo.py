# src/models/lstm_rul_mcdo.py
try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this notebook. Please install torch."
    ) from exc

class LSTMRULPredictorMCDropout(nn.Module):
    """
    LSTM-basierter RUL-Prädiktor mit Dropout für MC-Dropout-Unsicherheitsabschätzung.
    - Dropout wird sowohl in der LSTM-Schicht (zwischen den Layern) als auch
      vor der Fully-Connected-Schicht verwendet.
    - Für MC-Dropout wird das Modell auch im Inferenzmodus mit model.train() aufgerufen.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size=1, dropout_p=0.3):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_p if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # Dropout vor der FC-Schicht
        self.dropout = nn.Dropout(dropout_p)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)  # output: (batch, seq_len, hidden_size)

        # Letzter Zeitschritt
        last_output = output[:, -1, :]     # (batch, hidden_size)

        # Dropout auch im Forward – bei MC-Dropout später in model.train()-Mode aktiv
        last_output = self.dropout(last_output)

        prediction = self.fc(last_output)  # (batch, 1)
        return prediction
