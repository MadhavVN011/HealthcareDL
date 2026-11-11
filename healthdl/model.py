"""Multilayer neural network; outputs logits for CrossEntropyLoss."""

from torch import nn


class SymptomNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, output_size),
        )

    def forward(self, inputs):
        return self.layers(inputs)
