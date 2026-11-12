print("CLIENT STARTING")

import flwr as fl
import numpy as np
import os
import traceback

from universal_loader import load_dataset
from client_template import RetinaClient

# DATA PATH
DATA_PATHS = [
    "data/classification/gaussian_filtered_images",
    "../data/classification/gaussian_filtered_images",
]

data = None

for path in DATA_PATHS:
    if os.path.exists(path):
        print(f"Using dataset path: {path}")
        data = list(load_dataset("oct", path))
        break


# VALIDATION 1: DATA EXISTS
if data is None:
    raise FileNotFoundError("Dataset folder not found")

if len(data) == 0:
    raise ValueError("Dataset is EMPTY")

print("Raw samples loaded:", len(data))


# SPLIT DATA
X, y = zip(*data)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.int64)


# DEBUG CHECKS 
unique, counts = np.unique(y, return_counts=True)

print("CLASS DISTRIBUTION:", dict(zip(unique, counts)))
print("X shape:", X.shape)
print("y shape:", y.shape)


# HARD FAIL IF DATA IS BROKEN
if len(unique) < 2:
    raise ValueError(
        f"DATASET IS BROKEN: only one class found -> {unique}. "
        "Model will always predict same output."
    )


# FINAL VALIDATION
assert len(X) == len(y), "X and y mismatch"
assert X.ndim == 4, "Expected (N, C, H, W)"


# CREATE CLIENT
client = RetinaClient(X, y)

print("Client initialized successfully")


# START CLIENT
if __name__ == "__main__":
    try:
        fl.client.start_client(
            server_address="127.0.0.1:9090",
            client=client.to_client(),
        )

    except Exception:
        print("\n========== CLIENT CRASHED ==========\n")
        traceback.print_exc()