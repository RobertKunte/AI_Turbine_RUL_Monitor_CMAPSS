# src/models/world_model.py
from typing import Optional, Tuple, List

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

class WorldModelEncoderDecoderMultiTask(nn.Module):
    """
    Seq2Seq World Model mit zusätzlichem EOL-Head:
    - Encoder-Decoder LSTM, das eine RUL-Trajektorie vorhersagt.
    - Zusätzlich ein EOL-Head auf dem Encoder-Hidden-State,
      der einen einzelnen RUL-Punkt (z.B. RUL_{t+1}) vorhersagt.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.decoder = nn.LSTM(
            input_size=output_size,   # wir füttern RUL(t) als Input für RUL(t+1)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Lineares Mapping von hidden -> RUL (für die Trajektorie)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Zusätzlicher EOL-Head auf dem Encoder-Hidden-State (letzte Layer)
        self.eol_head =  nn.Sequential(
            nn.Linear(hidden_size, hidden_size),     # 1. Layer
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size// 2),  # 2. Layer
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),      # Output: Skalar-RUL
        )

    def forward(
        self,
        encoder_inputs: torch.Tensor,          # (B, L_past, F)
        decoder_targets: Optional[torch.Tensor] = None,  # (B, H, 1) oder None
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_inputs: (B, L_past, F)
            decoder_targets: (B, H, 1) oder None
            teacher_forcing_ratio: Wahrscheinlichkeit, mit GT zu füttern
            horizon: Anzahl der Future-Schritte; wenn None und decoder_targets ist nicht None:
                     horizon = decoder_targets.shape[1]

        Returns:
            traj_outputs: (B, H, 1) – vorhergesagte RUL-Trajektorie
            eol_pred:    (B, 1) – einzelner RUL-Punkt (Multi-Task-Head)
        """
        batch_size = encoder_inputs.size(0)

        # --- Encoder ---
        enc_out, (h_n, c_n) = self.encoder(encoder_inputs)
        # h_n: (num_layers, B, hidden_size)
        # Wir nehmen die letzte Layer als Zusammenfassung:
        enc_summary = h_n[-1]              # (B, hidden_size)

        # EOL-Vorhersage aus Encoder-Summary
        eol_pred = self.eol_head(enc_summary)   # (B, 1)

        # --- Decoder-Setup ---
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        if horizon is None:
            raise ValueError("Either decoder_targets or horizon must be provided.")

        # Initialer Decoder-Input:
        # Wir nehmen als Start-Token die erste True-RUL aus decoder_targets,
        # oder im reinen Inferenz-Fall die EOL-Prediction als Start.
        if decoder_targets is not None:
            # Teacher forcing initialisieren mit dem ersten Zielwert
            dec_input = decoder_targets[:, 0:1, :]   # (B, 1, 1)
        else:
            # Ohne Targets starten wir mit der EOL-Vorhersage
            dec_input = eol_pred.unsqueeze(1)        # (B, 1, 1)

        dec_hidden = (h_n, c_n)  # Encoder-Hidden-States als Start

        outputs = []
        for t in range(horizon):
            dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            step_pred = self.fc_out(dec_out)  # (B, 1, 1)

            outputs.append(step_pred)

            # Nächster Decoder-Input: Teacher Forcing oder Autoregressiv
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                # nächste GT-RUL
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                # autoregressiv
                dec_input = step_pred

        traj_outputs = torch.cat(outputs, dim=1)   # (B, H, 1)

        return traj_outputs, eol_pred


class WorldModelEncoderDecoderMultiTaskV9(nn.Module):
    """
    V9: Seq2Seq World Model with improved EOL head

    - Encoder: LSTM over past_len sensor+physics features
    - Decoder: LSTM that predicts a RUL trajectory of length `horizon`
    - EOL head:
        * takes the FULL encoder output over time
        * applies attention pooling over time dimension
        * then an MLP to predict a single scalar RUL (EOL-style)

    This is a drop-in replacement for the previous
    WorldModelEncoderDecoderMultiTask class: same forward signature.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # -------------------------
        # Decoder
        # -------------------------
        # We autoregress on the RUL itself: input_size = output_size (=1)
        self.decoder = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Map decoder hidden -> RUL (per time step)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # -------------------------
        # EOL Head: Attention pooling + MLP
        # -------------------------

        # Simple additive attention over encoder outputs:
        # scores_t = v^T * tanh(W * h_t)
        self.eol_attn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),   # score per time step
        )

        # MLP on the pooled encoder representation
        self.eol_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),  # scalar RUL
        )
        
        # V10.1: Use PyTorch's default initialization (no special bias/weight init)
        # This allows the model to learn from scratch without being biased towards a mean RUL
        # Previously: bias.fill_(90.0) and gain=0.1 caused collapse to constant predictions

    def forward(
        self,
        encoder_inputs: torch.Tensor,          # (B, L_past, F)
        decoder_targets: Optional[torch.Tensor] = None,  # (B, H, 1) or None
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args
        ----
        encoder_inputs : (B, L_past, F)
        decoder_targets : (B, H, 1) or None
            Ground-truth RUL trajectory. Used for teacher forcing and to
            determine `horizon` if not given.
        teacher_forcing_ratio : float
        horizon : int or None
            Number of future steps. If None and decoder_targets is given,
            horizon = decoder_targets.shape[1].

        Returns
        -------
        traj_outputs : (B, H, 1)
            Predicted RUL trajectory.
        eol_pred : (B, 1)
            Predicted scalar RUL at the evaluation point (EOL-style).
        """
        batch_size = encoder_inputs.size(0)

        # -------------------------
        # Encoder
        # -------------------------
        enc_out, (h_n, c_n) = self.encoder(encoder_inputs)
        # enc_out: (B, L_past, hidden_size)

        # -------------------------
        # EOL Head with Attention Pooling
        # -------------------------
        # Compute attention scores per time step
        # attn_scores: (B, L_past, 1)
        attn_scores = self.eol_attn(enc_out)  # (B, L_past, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L_past, 1) - softmax over time dimension

        # Weighted sum over time → pooled representation (B, hidden_size)
        # CRITICAL: Ensure gradients flow through attention pooling
        # enc_out: (B, L_past, hidden_size), attn_weights: (B, L_past, 1)
        # Broadcasting: (B, L_past, hidden_size) * (B, L_past, 1) -> (B, L_past, hidden_size)
        enc_pooled = (enc_out * attn_weights).sum(dim=1)  # (B, hidden_size)

        # EOL prediction
        eol_pred = self.eol_head(enc_pooled)  # (B, 1)

        # -------------------------
        # Decoder
        # -------------------------
        # Determine horizon
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        if horizon is None:
            raise ValueError("Either decoder_targets or horizon must be provided.")

        # Initial decoder input:
        if decoder_targets is not None:
            # use first GT RUL as start token
            dec_input = decoder_targets[:, 0:1, :]   # (B, 1, 1)
        else:
            # fall back to EOL prediction as start (in pure inference)
            dec_input = eol_pred.unsqueeze(1)        # (B, 1, 1)

        dec_hidden = (h_n, c_n)  # start from encoder hidden state

        outputs = []
        for t in range(horizon):
            dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            step_pred = self.fc_out(dec_out)  # (B, 1, 1)
            outputs.append(step_pred)

            # Teacher forcing or autoregressive feeding
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                dec_input = step_pred

        traj_outputs = torch.cat(outputs, dim=1)  # (B, H, 1)

        return traj_outputs, eol_pred


class WorldModelEncoderDecoderMultiTaskV11(nn.Module):
    """
    V11: World Model mit SEPARATEM EOL-Encoder.

    Wichtigste Änderung gegenüber V9/V10:
    - Trajektorien-Encoder und EOL-Encoder sind GETRENNT
    - Kein Shared-Encoder mehr, der für beide Tasks optimiert werden muss
    - Vermeidet Kollaps des EOL-Zweigs durch widersprüchliche Gradienten

    Architektur:
    - encoder_traj: LSTM für Trajektorien-Vorhersage
    - decoder_traj: LSTM-Decoder für Trajektorien
    - encoder_eol: SEPARATER LSTM für EOL-Vorhersage
    - eol_head: MLP auf letztem Hidden-State von encoder_eol
    """

    def __init__(
        self,
        input_size: int,
        hidden_size_traj: int = 64,
        hidden_size_eol: int = 64,
        num_layers_traj: int = 2,
        num_layers_eol: int = 2,
        output_size: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size_traj = hidden_size_traj
        self.hidden_size_eol = hidden_size_eol
        self.num_layers_traj = num_layers_traj
        self.num_layers_eol = num_layers_eol
        self.output_size = output_size

        # --- Trajektorien-Encoder/Decoder ---
        self.encoder_traj = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_traj,
            num_layers=num_layers_traj,
            batch_first=True,
            dropout=dropout if num_layers_traj > 1 else 0.0,
        )

        self.decoder_traj = nn.LSTM(
            input_size=output_size,
            hidden_size=hidden_size_traj,
            num_layers=num_layers_traj,
            batch_first=True,
            dropout=dropout if num_layers_traj > 1 else 0.0,
        )

        self.fc_traj = nn.Linear(hidden_size_traj, output_size)

        # --- Separater EOL-Encoder ---
        self.encoder_eol = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_eol,
            num_layers=num_layers_eol,
            batch_first=True,
            dropout=dropout if num_layers_eol > 1 else 0.0,
        )

        # EOL-Head: MLP auf letztem Hidden-State
        self.eol_head = nn.Sequential(
            nn.Linear(hidden_size_eol, hidden_size_eol),
            nn.ReLU(),
            nn.Linear(hidden_size_eol, hidden_size_eol // 2),
            nn.ReLU(),
            nn.Linear(hidden_size_eol // 2, 1),
        )

    def forward(
        self,
        encoder_inputs: torch.Tensor,  # (B, L, F)
        decoder_targets: Optional[torch.Tensor] = None,  # (B, H, 1) oder None
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass für Multi-Task World Model V11.

        Returns:
            traj_outputs: (B, H, 1) - Trajektorien-Vorhersagen
            eol_pred: (B, 1) - EOL-Vorhersage
        """
        B = encoder_inputs.size(0)

        # --- Trajektorienpfad ---
        enc_traj_out, (h_n_traj, c_n_traj) = self.encoder_traj(encoder_inputs)

        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        if horizon is None:
            raise ValueError("Either decoder_targets or horizon must be provided.")

        # Decoder-Input initialisieren
        if decoder_targets is not None:
            dec_input = decoder_targets[:, 0:1, :]  # (B, 1, 1)
        else:
            # Start token: z.B. letztes true RUL oder 0
            dec_input = torch.zeros(B, 1, 1, device=encoder_inputs.device)

        dec_hidden = (h_n_traj, c_n_traj)
        traj_outputs = []

        for t in range(horizon):
            dec_out, dec_hidden = self.decoder_traj(dec_input, dec_hidden)
            step_pred = self.fc_traj(dec_out)  # (B, 1, 1)
            traj_outputs.append(step_pred)

            # Teacher Forcing
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                dec_input = step_pred

        traj_outputs = torch.cat(traj_outputs, dim=1)  # (B, H, 1)

        # --- EOL-Pfad (separat) ---
        enc_eol_out, (h_n_eol, c_n_eol) = self.encoder_eol(encoder_inputs)
        h_last_eol = h_n_eol[-1]  # (B, hidden_size_eol)
        eol_pred = self.eol_head(h_last_eol)  # (B, 1)

        return traj_outputs, eol_pred


class WorldModelEncoderDecoderUniversalV2(nn.Module):
    """
    World Model using UniversalEncoderV2 as encoder.
    
    Phase 4: Uses UniversalEncoderV2 (multi-scale CNN + Transformer) as encoder
    for seq2seq RUL trajectory prediction with EOL head.
    
    Architecture:
    - encoder: UniversalEncoderV2 (multi-scale CNN + Transformer/LSTM)
    - decoder: LSTM decoder for RUL trajectory
    - eol_head: MLP on encoder output for EOL prediction
    
    This is designed for Phase 4 residual feature experiments (464 features).
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 96,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        output_size: int = 1,
        num_conditions: Optional[int] = None,
        cond_emb_dim: int = 4,
        kernel_sizes: List[int] = None,
        seq_encoder_type: str = "transformer",
        use_layer_norm: bool = True,
        max_seq_len: int = 300,
        decoder_hidden_size: Optional[int] = None,
        decoder_num_layers: int = 2,
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]
        
        if decoder_hidden_size is None:
            decoder_hidden_size = d_model
        
        self.input_size = input_size
        self.d_model = d_model
        self.output_size = output_size
        self.num_conditions = num_conditions
        self.decoder_num_layers = decoder_num_layers
        
        # Import UniversalEncoderV2
        from .universal_encoder_v1 import UniversalEncoderV2
        
        # Encoder: UniversalEncoderV2
        self.encoder = UniversalEncoderV2(
            input_dim=input_size,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
            max_seq_len=max_seq_len,
        )
        
        # Decoder: LSTM for trajectory prediction
        self.decoder = nn.LSTM(
            input_size=output_size,  # Autoregressive: feed previous RUL prediction
            hidden_size=decoder_hidden_size,
            num_layers=decoder_num_layers,
            batch_first=True,
            dropout=dropout if decoder_num_layers > 1 else 0.0,
        )
        
        # Project encoder output to decoder initial hidden state
        self.encoder_to_decoder_h = nn.Linear(d_model, decoder_hidden_size * decoder_num_layers)
        self.encoder_to_decoder_c = nn.Linear(d_model, decoder_hidden_size * decoder_num_layers)
        
        # Decoder output projection
        self.fc_out = nn.Linear(decoder_hidden_size, output_size)
        
        # EOL Head: MLP on encoder output
        self.eol_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,  # (B, L_past, F)
        decoder_targets: Optional[torch.Tensor] = None,  # (B, H, 1) or None
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,  # (B,) - condition IDs for encoder
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            encoder_inputs: (B, L_past, F) - Past sequence
            decoder_targets: (B, H, 1) or None - Future RUL trajectory (for teacher forcing)
            teacher_forcing_ratio: Probability of using ground truth in decoder
            horizon: Number of future steps (if decoder_targets is None)
            cond_ids: (B,) - Condition IDs for encoder (required if num_conditions > 1)
        
        Returns:
            traj_outputs: (B, H, 1) - Predicted RUL trajectory
            eol_pred: (B, 1) - Predicted EOL RUL
        """
        B = encoder_inputs.size(0)
        
        # --- Encoder ---
        # UniversalEncoderV2 returns (B, d_model) - sequence embedding
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)  # (B, d_model)
        
        # EOL prediction from encoder embedding
        eol_pred = self.eol_head(enc_emb)  # (B, 1)
        
        # --- Decoder Setup ---
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        if horizon is None:
            raise ValueError("Either decoder_targets or horizon must be provided.")
        
        # Initialize decoder hidden state from encoder embedding
        h_0 = self.encoder_to_decoder_h(enc_emb)  # (B, decoder_hidden_size * decoder_num_layers)
        c_0 = self.encoder_to_decoder_c(enc_emb)  # (B, decoder_hidden_size * decoder_num_layers)
        
        # Reshape for LSTM: (num_layers, B, hidden_size)
        h_0 = h_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        c_0 = c_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
        
        dec_hidden = (h_0, c_0)
        
        # Initial decoder input
        if decoder_targets is not None:
            dec_input = decoder_targets[:, 0:1, :]  # (B, 1, 1)
        else:
            dec_input = eol_pred.unsqueeze(1)  # (B, 1, 1)
        
        # --- Decoder Loop ---
        traj_outputs = []
        for t in range(horizon):
            dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
            step_pred = self.fc_out(dec_out)  # (B, 1, 1)
            traj_outputs.append(step_pred)
            
            # Teacher forcing or autoregressive
            if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                if t + 1 < decoder_targets.size(1):
                    dec_input = decoder_targets[:, t + 1 : t + 2, :]
                else:
                    dec_input = step_pred
            else:
                dec_input = step_pred
        
        traj_outputs = torch.cat(traj_outputs, dim=1)  # (B, H, 1)
        
        return traj_outputs, eol_pred


class WorldModelUniversalV3(nn.Module):
    """
    Universal World Model v3 with Health Index Head.
    
    Phase 5: Enhanced World Model with:
    - UniversalEncoderV2 (multi-scale CNN + Transformer) as encoder
    - LSTM Decoder for trajectory prediction (horizon=40)
    - Three heads:
        * Trajectory-Head: Predicts HI-proxy trajectory over horizon
        * EOL-Head: Predicts EOL RUL (strongly weighted)
        * HI-Head: Predicts Health Index in [0, 1] (with monotonicity)
    
    This is designed for Phase 5 residual feature experiments (464 features).
    Uses same architecture pattern as RULHIUniversalModelV2 for compatibility.
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 96,
        num_layers: int = 3,
        nhead: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.1,
        num_conditions: Optional[int] = None,
        cond_emb_dim: int = 4,
        kernel_sizes: List[int] = None,
        seq_encoder_type: str = "transformer",
        use_layer_norm: bool = True,
        max_seq_len: int = 300,
        decoder_hidden_size: Optional[int] = None,
        decoder_num_layers: int = 2,
        horizon: int = 40,
        # v3 extensions: HI fusion into EOL head
        use_hi_in_eol: bool = False,
        use_hi_slope_in_eol: bool = False,
        # Decoder type selection
        decoder_type: str = "lstm",  # "lstm" (default) or "tf_ar" or "tf_ar_xattn"
    ):
        super().__init__()
        
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 9]
        
        if decoder_hidden_size is None:
            decoder_hidden_size = d_model
        
        self.input_size = input_size
        self.d_model = d_model
        self.num_conditions = num_conditions
        self.decoder_num_layers = decoder_num_layers
        self.horizon = horizon
        self.use_hi_in_eol = use_hi_in_eol
        self.use_hi_slope_in_eol = use_hi_slope_in_eol
        self.decoder_type = decoder_type
        
        # Import UniversalEncoderV2
        from .universal_encoder_v1 import UniversalEncoderV2
        
        # Encoder: UniversalEncoderV2 (same as v2)
        self.encoder = UniversalEncoderV2(
            input_dim=input_size,
            d_model=d_model,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            num_conditions=num_conditions,
            cond_emb_dim=cond_emb_dim,
            kernel_sizes=kernel_sizes,
            seq_encoder_type=seq_encoder_type,
            use_layer_norm=use_layer_norm,
            max_seq_len=max_seq_len,
        )
        
        # Decoder: LSTM (default) or Transformer AR
        if decoder_type == "lstm":
            # LSTM decoder (default, unchanged)
            self.decoder = nn.LSTM(
                input_size=1,  # Autoregressive: feed previous prediction
                hidden_size=decoder_hidden_size,
                num_layers=decoder_num_layers,
                batch_first=True,
                dropout=dropout if decoder_num_layers > 1 else 0.0,
            )
            
            # Project encoder output to decoder initial hidden state
            self.encoder_to_decoder_h = nn.Linear(d_model, decoder_hidden_size * decoder_num_layers)
            self.encoder_to_decoder_c = nn.Linear(d_model, decoder_hidden_size * decoder_num_layers)
            self.tf_decoder = None
        elif decoder_type in {"tf_ar", "tf_ar_xattn"}:
            # Transformer AR decoder
            from .decoders.transformer_ar_decoder import TransformerARDecoder
            
            use_cross_attention = (decoder_type == "tf_ar_xattn")
            self.tf_decoder = TransformerARDecoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=decoder_num_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                horizon=horizon,
                use_cross_attention=use_cross_attention,
            )
            # LSTM decoder components not used, but keep for compatibility
            self.decoder = None
            self.encoder_to_decoder_h = None
            self.encoder_to_decoder_c = None
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}. Must be 'lstm', 'tf_ar', or 'tf_ar_xattn'")
        
        # Shared head (like RULHIUniversalModelV2) for HI and (base) EOL features
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Trajectory-Head: Predicts HI-proxy trajectory from decoder output
        # For LSTM: maps decoder hidden -> scalar
        # For Transformer: output_head is inside TransformerARDecoder
        if decoder_type == "lstm":
            self.traj_head = nn.Linear(decoder_hidden_size, 1)
        else:
            self.traj_head = None  # Transformer decoder has its own output head
        
        # Determine number of additional HI features for EOL fusion
        hi_feat_dim = 0
        if self.use_hi_in_eol:
            hi_feat_dim += 1  # hi_current
            if self.use_hi_slope_in_eol:
                hi_feat_dim += 1  # hi_slope
        
        # EOL-Head: MLP/Linear on shared encoder features (+ optional HI fusion)
        eol_in_dim = d_model + hi_feat_dim if self.use_hi_in_eol else d_model
        self.fc_rul = nn.Linear(eol_in_dim, 1)
        
        # HI-Head: Predicts Health Index in [0, 1] (with monotonicity)
        self.fc_health = nn.Linear(d_model, 1)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        encoder_inputs: torch.Tensor,  # (B, L_past, F)
        decoder_targets: Optional[torch.Tensor] = None,  # (B, H, 1) or None
        teacher_forcing_ratio: float = 0.5,
        horizon: Optional[int] = None,
        cond_ids: Optional[torch.Tensor] = None,  # (B,) - condition IDs for encoder
    ) -> dict:
        """
        Forward pass.
        
        Args:
            encoder_inputs: (B, L_past, F) - Past sequence
            decoder_targets: (B, H, 1) or None - Future trajectory (for teacher forcing)
            teacher_forcing_ratio: Probability of using ground truth in decoder
            horizon: Number of future steps (if decoder_targets is None, uses self.horizon)
            cond_ids: (B,) - Condition IDs for encoder (required if num_conditions > 1)
        
        Returns:
            Dictionary with:
                "traj": (B, H, 1) - Predicted HI-proxy trajectory
                "eol": (B, 1) - Predicted EOL RUL
                "hi": (B, 1) - Predicted Health Index in [0, 1]
        """
        B = encoder_inputs.size(0)
        
        # --- Encoder ---
        # UniversalEncoderV2 returns (B, d_model) - sequence embedding
        enc_emb = self.encoder(encoder_inputs, cond_ids=cond_ids)  # (B, d_model)
        
        # Shared features for EOL and HI heads
        h_shared = self.shared_head(enc_emb)  # (B, d_model)
        
        # HI prediction from shared features (sigmoid to [0, 1])
        hi_logit = self.fc_health(h_shared)  # (B, 1)
        hi_pred = torch.sigmoid(hi_logit)  # (B, 1)
        
        # --- Decoder Setup ---
        if decoder_targets is not None and horizon is None:
            horizon = decoder_targets.size(1)
        elif horizon is None:
            horizon = self.horizon
        
        # Decoder forward: LSTM or Transformer AR
        if self.decoder_type == "lstm":
            # LSTM decoder path (original, unchanged)
            # Initialize decoder hidden state from encoder embedding
            h_0 = self.encoder_to_decoder_h(enc_emb)  # (B, decoder_hidden_size * decoder_num_layers)
            c_0 = self.encoder_to_decoder_c(enc_emb)  # (B, decoder_hidden_size * decoder_num_layers)
            
            # Reshape for LSTM: (num_layers, B, hidden_size)
            h_0 = h_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
            c_0 = c_0.view(B, self.decoder_num_layers, -1).transpose(0, 1).contiguous()
            
            dec_hidden = (h_0, c_0)
            
            # Initial decoder input:
            # - If we have decoder_targets (training/teacher forcing), start from first target step.
            # - Otherwise:
            #     * In HI-fusion mode: start from HI prediction (consistent with HI trajectory)
            #     * In base mode: we will start from a simple EOL estimate later if needed.
            if decoder_targets is not None:
                dec_input = decoder_targets[:, 0:1, :]  # (B, 1, 1)
            else:
                if self.use_hi_in_eol:
                    dec_input = hi_pred.unsqueeze(1)  # (B, 1, 1)
                else:
                    # Base mode without HI fusion: start from zeros (decoder learns its own dynamics)
                    dec_input = torch.zeros(B, 1, 1, device=encoder_inputs.device)
            
            # --- Decoder Loop ---
            traj_outputs = []
            for t in range(horizon):
                dec_out, dec_hidden = self.decoder(dec_input, dec_hidden)
                step_pred = self.traj_head(dec_out)  # (B, 1, 1) - HI-proxy prediction
                traj_outputs.append(step_pred)
                
                # Teacher forcing or autoregressive
                if decoder_targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    if t + 1 < decoder_targets.size(1):
                        dec_input = decoder_targets[:, t + 1 : t + 2, :]
                    else:
                        dec_input = step_pred
                else:
                    dec_input = step_pred
            
            traj_outputs = torch.cat(traj_outputs, dim=1)  # (B, H, 1)
            
        else:
            # Transformer AR decoder path
            # Determine mode
            mode = "train" if decoder_targets is not None else "inference"
            
            # For cross-attention variant, we would need encoder sequence
            # But UniversalEncoderV2 returns only (B, d_model), not sequence
            # So we pass None for enc_seq (decoder will use enc_token as fallback)
            enc_seq = None
            
            # Call transformer decoder
            traj_outputs = self.tf_decoder(
                enc_token=enc_emb,  # (B, d_model)
                y_teacher=decoder_targets,  # (B, H, 1) or None
                enc_seq=enc_seq,  # None (encoder doesn't provide sequence)
                cond_ctx=None,  # Reserved for future use
                mode=mode,
            )  # (B, H, 1)
        hi_seq = traj_outputs.squeeze(-1)  # (B, H)

        # --- EOL Head with optional HI fusion ---
        if self.use_hi_in_eol:
            # Use HI sequence to build EOL features: current HI + optional local slope
            hi_current = hi_seq[:, 0].unsqueeze(-1)  # (B, 1)
            if hi_seq.size(1) > 1:
                k = min(3, hi_seq.size(1) - 1)
                hi_slope = (hi_seq[:, 0] - hi_seq[:, k]).unsqueeze(-1)  # (B, 1)
            else:
                hi_slope = torch.zeros_like(hi_current)

            eol_features = [h_shared, hi_current]
            if self.use_hi_slope_in_eol:
                eol_features.append(hi_slope)

            eol_input = torch.cat(eol_features, dim=-1)  # (B, d_model + n_hi_features)
            eol_pred = self.fc_rul(eol_input)  # (B, 1)
        else:
            # Base mode: EOL depends only on shared encoder features (backwards-compatible)
            eol_pred = self.fc_rul(h_shared)  # (B, 1)
        
        return {
            "traj": traj_outputs,  # (B, H, 1)
            "eol": eol_pred,       # (B, 1)
            "hi": hi_pred,         # (B, 1)
        }
