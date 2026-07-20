import torch
import torch.nn as nn


class CNNBiLSTM(nn.Module):
    """1D-CNN-BiLSTM for network intrusion detection (paper Table 1).

    Input  : (B, 78)          — 78 normalised features per sample
    Output : (B, num_classes) — raw logits, no softmax

    Architecture:
        (B,78) → unsqueeze → (B,1,78)
        Conv1D(1→64,  k=3, same) → BN → ReLU → MaxPool/2   → (B, 64, 39)
        Conv1D(64→128,k=3, same) → BN → ReLU → MaxPool/2   → (B,128, 19)
        permute → (B,19,128)
        BiLSTM(hidden=64, bidirectional) → last-step cat    → (B,128)
        Dropout(0.5) → Linear(128→64) → ReLU → Linear(64→C)
    """

    def __init__(self, input_dim: int = 78, num_classes: int = 15):
        super().__init__()
        # Conv block 1
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)   # padding=1 → 'same' for k=3
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        # Conv block 2
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        # BiLSTM: takes (B, seq=19, feat=128) → h_n is (2, B, 64)
        self.bilstm  = nn.LSTM(128, 64, batch_first=True, bidirectional=True)
        # Classifier head
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(128, 64)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, 78)
        x = x.unsqueeze(1)                          # (B, 1, 78)

        x = self.relu(self.bn1(self.conv1(x)))      # (B, 64, 78)
        x = self.pool1(x)                           # (B, 64, 39)

        x = self.relu(self.bn2(self.conv2(x)))      # (B, 128, 39)
        x = self.pool2(x)                           # (B, 128, 19)

        x = x.permute(0, 2, 1)                     # (B, 19, 128)
        _, (h_n, _) = self.bilstm(x)               # h_n: (2, B, 64)
        x = torch.cat([h_n[0], h_n[1]], dim=1)     # (B, 128) — forward + backward last states

        x = self.dropout(x)
        x = self.relu(self.fc1(x))                  # (B, 64)
        return self.fc2(x)                          # (B, num_classes)


class CNNOnly(nn.Module):
    """CNN-only ablation: same two conv blocks as CNNBiLSTM, BiLSTM replaced by
    AdaptiveAvgPool1d(1).  Classifier head (Dropout→128→64→C) is identical."""

    def __init__(self, input_dim: int = 78, num_classes: int = 15):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(128, 64)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.gap(x).squeeze(-1)       # (B, 128)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class BiLSTMOnly(nn.Module):
    """BiLSTM-only ablation: no CNN; each of the 78 scalar features is one
    time-step (dim=1) fed directly into BiLSTM.  Classifier head identical."""

    def __init__(self, input_dim: int = 78, num_classes: int = 15):
        super().__init__()
        self.bilstm  = nn.LSTM(1, 64, batch_first=True, bidirectional=True)
        self.relu    = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1     = nn.Linear(128, 64)
        self.fc2     = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)               # (B, 78, 1)
        _, (h_n, _) = self.bilstm(x)      # h_n: (2, B, 64)
        x = torch.cat([h_n[0], h_n[1]], dim=1)   # (B, 128)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def count_params(model: nn.Module) -> int:
    """Print and return the number of trainable parameters."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters    : {total:,}")
    print(f"  Trainable parameters: {trainable:,}")
    return trainable


if __name__ == "__main__":
    m = CNNBiLSTM()
    x = torch.randn(4, 78)
    print("Input :", x.shape)
    print("Output:", m(x).shape)
    count_params(m)
