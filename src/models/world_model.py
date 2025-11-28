# src/models/world_model.py
from typing import Optional, Tuple

try:
    import torch  # type: ignore[import]
    import torch.nn as nn  # type: ignore[import]
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for this notebook. Please install torch."
    ) from exc


class WorldModelEncoderDecoder(nn.Module):
    """
    Encoder-Decoder LSTM world model.

    - Encoder: consumes past sensor + physics features.
    - Decoder: predicts future sequences (e.g. future sensors or future RUL).
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # Encoder LSTM
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder LSTM
        self.decoder = nn.LSTM(
            input_size=output_size,  # we feed back previous prediction (or teacher forcing)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map decoder hidden state -> output features
        self.output_projection = nn.Linear(hidden_size, output_size)

    def forward(
        self,
        encoder_inputs: torch.Tensor,   # (B, L_past, input_size)
        decoder_targets: Optional[torch.Tensor] = None,  # (B, H, output_size) for teacher forcing
        horizon: Optional[int] = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """
        Returns:
            outputs: (B, H, output_size)
        """
        batch_size = encoder_inputs.size(0)

        # ---- Encode past ----
        encoder_outputs, (h, c) = self.encoder(encoder_inputs)

        if decoder_targets is not None:
            # use length of target sequence as horizon
            H = decoder_targets.size(1)
        elif horizon is not None:
            H = horizon
        else:
            raise ValueError("Either decoder_targets or horizon must be provided.")

        # ---- Decode future ----
        outputs = []
        # init first decoder input: either first target (teacher forcing) or zeros
        if decoder_targets is not None:
            # use first ground truth as starting token
            decoder_input = decoder_targets[:, 0:1, :]  # (B, 1, output_size)
        else:
            decoder_input = torch.zeros(batch_size, 1, self.output_size, device=encoder_inputs.device)

        h_dec, c_dec = h, c

        for t in range(H):
            dec_out, (h_dec, c_dec) = self.decoder(decoder_input, (h_dec, c_dec))
            step_output = self.output_projection(dec_out)  # (B, 1, output_size)
            outputs.append(step_output)

            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio and t + 1 < H:
                # Next input is ground-truth
                decoder_input = decoder_targets[:, t + 1 : t + 2, :]
            else:
                # Next input is model prediction
                decoder_input = step_output

        outputs = torch.cat(outputs, dim=1)  # (B, H, output_size)
        return outputs
