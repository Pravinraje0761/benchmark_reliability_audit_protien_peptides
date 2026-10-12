"""CNN and BiLSTM baselines (M05, M06).

Isolated in their own module so the classical sweep never imports torch. These
are the GPU-bound pipelines; run them last, per the staging plan.
"""
from __future__ import annotations


def build_sequence_model(kind: str, task: str, seed: int):
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    out_dim = 1

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(20, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
                nn.AdaptiveMaxPool1d(1), nn.Flatten(),
                nn.Dropout(0.3), nn.Linear(128, out_dim),
            )

        def forward(self, x):            # x: (B, L, 20)
            return self.net(x.transpose(1, 2)).squeeze(-1)

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(20, 64, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, out_dim))

        def forward(self, x):
            h, _ = self.rnn(x)
            return self.head(h.mean(dim=1)).squeeze(-1)

    return (CNN() if kind == "cnn" else BiLSTM())
