"""Small inference-only execution of the trained network's exported dense layers.

Dropout is inactive at inference. Matrix multiplication and ReLU execute the same
131 -> 128 -> 64 -> 41 network without a training framework in the web function.
"""

from pathlib import Path

import numpy as np


class NumpyNetwork:
    def __init__(self, path: Path, input_size: int, output_size: int):
        expected = {"w0": (128, input_size), "b0": (128,), "w1": (64, 128),
                    "b1": (64,), "w2": (output_size, 64), "b2": (output_size,)}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ValueError("Invalid neural network export")
            self.weights = {name: archive[name].astype(np.float32) for name in expected}
        if any(self.weights[name].shape != shape or not np.isfinite(self.weights[name]).all()
               for name, shape in expected.items()):
            raise ValueError("Invalid neural network parameter dimensions or values")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        values = np.asarray(inputs, dtype=np.float32)
        for layer in range(3):
            values = values @ self.weights[f"w{layer}"].T + self.weights[f"b{layer}"]
            if layer < 2:
                values = np.maximum(values, 0)
        return values

    def probabilities(self, inputs: np.ndarray) -> np.ndarray:
        logits = self.forward(inputs)
        values = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return values / values.sum(axis=-1, keepdims=True)
