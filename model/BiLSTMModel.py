import torch
import torch.nn as nn


class BiLSTMChordTagger(nn.Module):
    """
    A BiLSTM-based sequence tagger that predicts chords from symbolic melody (pitch) input.

    Args:
        n_pitch_tokens (int): Size of the pitch vocabulary (incl. <PAD> and 'rest').
        n_chord_tokens (int): Size of the chord vocabulary (incl. <PAD> and 'N.C.').
        embed_dim (int): Dimension of embedding vectors.
        lstm_hidden (int): Number of hidden units in LSTM.
        lstm_layers (int): Number of stacked LSTM layers.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_pitch_tokens: int,
        n_chord_tokens: int,
        embed_dim: int = 128,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()

        # Embedding layer for pitch sequences
        self.embedding = nn.Embedding(
            num_embeddings=n_pitch_tokens,
            embedding_dim=embed_dim,
            padding_idx=pitch2idx["<PAD>"]
        )

        # Bidirectional LSTM encoder
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        # Output layer: maps hidden states to chord vocabulary
        self.classifier = nn.Linear(
            in_features=lstm_hidden * 2,  # because bidirectional
            out_features=n_chord_tokens
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_pitches: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            input_pitches (LongTensor): (B, L) Padded pitch index sequences
            lengths (LongTensor): (B,) Original sequence lengths

        Returns:
            logits (FloatTensor): (B, L, n_chord_tokens) Unnormalized chord predictions
        """
        # Embedding
        emb = self.embedding(input_pitches)        # (B, L, embed_dim)
        emb = self.dropout(emb)

        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_out, _ = self.lstm(packed)

        # Unpack sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True
        )  # (B, L, lstm_hidden * 2)

        # Classification
        logits = self.classifier(out)  # (B, L, n_chord_tokens)

        return logits
