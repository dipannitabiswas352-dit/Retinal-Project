# ============================================================
# SHARED-BACKBONE FEDERATED RETINA CLIENT
# ============================================================

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

from torch.utils.data import Dataset, DataLoader, random_split

from model import Net


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
FEDPROX_MU = 0.01

TRAIN_RATIO = 0.80
RANDOM_SEED = 42


# ============================================================
# LAZY NUMPY DATASET
# ============================================================

class RetinaDataset(Dataset):

    def __init__(self, X, y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.y)

    def __getitem__(self, index):

        image = self.X[index]
        label = self.y[index]

        image = np.asarray(image)

        # HWC -> CHW
        if image.ndim == 3 and image.shape[-1] == 3:

            image = np.transpose(
                image,
                (2, 0, 1)
            )

        image = np.ascontiguousarray(
            image
        )

        image = torch.from_numpy(
            image
        ).float()

        # Normalize uint8 images
        if image.max() > 1.0:

            image = image / 255.0

        # ImageNet normalization
        mean = torch.tensor(
            [0.485, 0.456, 0.406],
            dtype=torch.float32
        ).view(3, 1, 1)

        std = torch.tensor(
            [0.229, 0.224, 0.225],
            dtype=torch.float32
        ).view(3, 1, 1)

        image = (
            image - mean
        ) / std

        label = torch.tensor(
            int(label),
            dtype=torch.long
        )

        return image, label


# ============================================================
# SHARED PARAMETERS
#
# ONLY TRAINABLE BACKBONE PARAMETERS ARE FEDERATED
# ============================================================

def get_shared_parameters(model):

    parameters = []

    for name, parameter in model.named_parameters():

        # Only backbone
        if not name.startswith("backbone."):

            continue

        # Only trainable backbone parameters
        if not parameter.requires_grad:

            continue

        parameters.append(
            parameter.detach()
            .cpu()
            .numpy()
            .copy()
        )

    return parameters


# ============================================================
# SET SHARED PARAMETERS
#
# ONLY TRAINABLE BACKBONE PARAMETERS ARE LOADED
# ============================================================

def set_shared_parameters(
    model,
    parameters
):

    trainable_parameters = []

    for name, parameter in model.named_parameters():

        # Only backbone
        if not name.startswith("backbone."):

            continue

        # Ignore frozen parameters
        if not parameter.requires_grad:

            continue

        trainable_parameters.append(
            (name, parameter)
        )

    # --------------------------------------------------------
    # PARAMETER COUNT CHECK
    # --------------------------------------------------------

    if len(parameters) != len(
        trainable_parameters
    ):

        raise ValueError(
            "Shared parameter count mismatch: "
            f"received {len(parameters)}, "
            f"expected {len(trainable_parameters)}"
        )

    # --------------------------------------------------------
    # LOAD PARAMETERS
    # --------------------------------------------------------

    with torch.no_grad():

        for (
            name,
            parameter
        ), received_value in zip(
            trainable_parameters,
            parameters
        ):

            received_array = np.asarray(
                received_value
            )

            # ------------------------------------------------
            # SHAPE CHECK
            # ------------------------------------------------

            if tuple(
                received_array.shape
            ) != tuple(
                parameter.shape
            ):

                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"received "
                    f"{received_array.shape}, "
                    f"expected "
                    f"{tuple(parameter.shape)}"
                )

            # ------------------------------------------------
            # COPY GLOBAL VALUE
            # ------------------------------------------------

            parameter.copy_(
                torch.from_numpy(
                    received_array
                ).to(
                    device=parameter.device,
                    dtype=parameter.dtype
                )
            )


# ============================================================
# RETINA CLIENT
# ============================================================

class RetinaClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        X,
        y,
        num_classes=None
    ):

        print()
        print("=" * 60)
        print("INITIALIZING RETINA CLIENT")
        print("=" * 60)

        # ====================================================
        # DEVICE
        # ====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            "Device:",
            self.device
        )

        # ====================================================
        # LABELS
        # ====================================================

        y_array = np.asarray(
            y,
            dtype=np.int64
        )

        unique_labels = np.unique(
            y_array
        )

        if num_classes is None:

            num_classes = int(
                np.max(unique_labels)
            ) + 1

        self.num_classes = int(
            num_classes
        )

        print(
            "Number of classes:",
            self.num_classes
        )

        # ====================================================
        # MODEL
        # ====================================================

        self.model = Net(
            num_classes=self.num_classes
        ).to(
            self.device
        )

        print(
            "Using shared-backbone model"
        )

        self.criterion = nn.CrossEntropyLoss()

        # ====================================================
        # DATA VALIDATION
        # ====================================================

        if len(X) != len(y_array):

            raise ValueError(
                f"Number of images ({len(X)}) "
                f"does not match labels ({len(y_array)})"
            )

        if len(X) < 2:

            raise ValueError(
                "Dataset must contain at least 2 samples."
            )

        if len(unique_labels) < 2:

            raise ValueError(
                f"Dataset contains only one class: "
                f"{unique_labels}"
            )

        if np.any(
            y_array < 0
        ):

            raise ValueError(
                f"Negative labels detected: "
                f"{unique_labels}"
            )

        if np.any(
            y_array >= self.num_classes
        ):

            raise ValueError(
                f"Invalid labels detected: "
                f"{unique_labels}. "
                f"Expected 0-{self.num_classes - 1}."
            )

        # ====================================================
        # DATASET
        # ====================================================

        dataset = RetinaDataset(
            X,
            y_array
        )

        # ====================================================
        # TRAIN / VALIDATION SPLIT
        # ====================================================

        train_size = int(
            TRAIN_RATIO * len(dataset)
        )

        val_size = (
            len(dataset)
            - train_size
        )

        if train_size < 1:

            raise ValueError(
                "Training set would be empty."
            )

        if val_size < 1:

            raise ValueError(
                "Validation set would be empty."
            )

        generator = torch.Generator().manual_seed(
            RANDOM_SEED
        )

        train_set, val_set = random_split(
            dataset,
            [train_size, val_size],
            generator=generator
        )

        # ====================================================
        # DATALOADERS
        # ====================================================

        self.train_loader = DataLoader(
            train_set,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            )
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=(
                self.device.type == "cuda"
            )
        )

        self.dataset_size = len(
            train_set
        )

        # ====================================================
        # FEDERATED PARAMETER CHECK
        # ====================================================

        shared_parameters = get_shared_parameters(
            self.model
        )

        print()
        print(
            "Training samples  :",
            train_size
        )

        print(
            "Validation samples:",
            val_size
        )

        print(
            "Total samples     :",
            len(dataset)
        )

        print()
        print(
            "Federated tensors:",
            len(shared_parameters)
        )

        print(
            "Federated component : "
            "TRAINABLE BACKBONE ONLY"
        )

        print(
            "Local component     : "
            f"{self.num_classes}-CLASS HEAD"
        )

        print("=" * 60)
        print("RETINA CLIENT READY")
        print("=" * 60)

    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config=None
    ):

        parameters = get_shared_parameters(
            self.model
        )

        print(
            "Sending shared tensors:",
            len(parameters)
        )

        return parameters

    # ========================================================
    # SET PARAMETERS
    # ========================================================

    def set_parameters(
        self,
        parameters
    ):

        print(
            "Loading global shared parameters..."
        )

        set_shared_parameters(
            self.model,
            parameters
        )

        print(
            "Global shared parameters loaded."
        )

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        print()
        print("=" * 60)
        print(">>> FIT STARTED")
        print("=" * 60)

        start_time = time.time()

        # ====================================================
        # RECEIVE GLOBAL BACKBONE
        # ====================================================

        self.set_parameters(
            parameters
        )

        # ====================================================
        # SNAPSHOT GLOBAL TRAINABLE BACKBONE
        # ====================================================

        global_backbone = []

        for name, parameter in (
            self.model.named_parameters()
        ):

            if not name.startswith(
                "backbone."
            ):

                continue

            if not parameter.requires_grad:

                continue

            global_backbone.append(
                parameter.detach()
                .clone()
                .to(self.device)
            )

        # ====================================================
        # OPTIMIZER
        #
        # Trainable parameters only
        # ====================================================

        optimizer = optim.Adam(
            filter(
                lambda parameter:
                parameter.requires_grad,
                self.model.parameters()
            ),
            lr=LEARNING_RATE
        )

        self.model.train()

        # ====================================================
        # METRICS
        # ====================================================

        total_loss = 0.0
        correct = 0
        total = 0

        # ====================================================
        # LOCAL TRAINING
        # ====================================================

        num_batches = len(
            self.train_loader
        )

        print(
            "Training batches:",
            num_batches
        )

        for batch_idx, (
            X_batch,
            y_batch
        ) in enumerate(
            self.train_loader,
            start=1
        ):

            X_batch = X_batch.to(
                self.device
            )

            y_batch = y_batch.to(
                self.device
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            outputs = self.model(
                X_batch
            )

            ce_loss = self.criterion(
                outputs,
                y_batch
            )

            # ------------------------------------------------
            # FEDPROX
            # ------------------------------------------------

            prox_loss = torch.tensor(
                0.0,
                device=self.device
            )

            trainable_backbone = []

            for name, parameter in (
                self.model.named_parameters()
            ):

                if not name.startswith(
                    "backbone."
                ):

                    continue

                if not parameter.requires_grad:

                    continue

                trainable_backbone.append(
                    parameter
                )

            for (
                global_parameter,
                local_parameter
            ) in zip(
                global_backbone,
                trainable_backbone
            ):

                prox_loss = (
                    prox_loss
                    + torch.sum(
                        (
                            local_parameter
                            - global_parameter
                        ) ** 2
                    )
                )

            loss = (
                ce_loss
                + (
                    FEDPROX_MU / 2.0
                ) * prox_loss
            )

            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            batch_size = (
                y_batch.size(0)
            )

            total_loss += (
                ce_loss.item()
                * batch_size
            )

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                (
                    predicted == y_batch
                )
                .sum()
                .item()
            )

            total += batch_size

            # ------------------------------------------------
            # PROGRESS
            # ------------------------------------------------

            if (
                batch_idx % 10 == 0
                or batch_idx == num_batches
            ):

                print(
                    f"Batch "
                    f"{batch_idx}/"
                    f"{num_batches} | "
                    f"Loss: "
                    f"{ce_loss.item():.4f}"
                )

        # ====================================================
        # FINAL METRICS
        # ====================================================

        avg_loss = (
            total_loss / total
            if total > 0
            else 0.0
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        elapsed = (
            time.time()
            - start_time
        )

        print()
        print(
            "LOCAL TRAINING COMPLETE"
        )

        print(
            f"Training time : {elapsed:.2f}s"
        )

        print(
            f"Loss          : {avg_loss:.4f}"
        )

        print(
            f"Accuracy      : {accuracy:.4f}"
        )

        # ====================================================
        # RETURN SHARED BACKBONE
        # ====================================================

        shared_parameters = (
            get_shared_parameters(
                self.model
            )
        )

        print(
            "Sending shared tensors:",
            len(shared_parameters)
        )

        print(
            "Sending TRAINABLE BACKBONE only."
        )

        return (
            shared_parameters,
            self.dataset_size,
            {
                "loss": float(
                    avg_loss
                ),
                "accuracy": float(
                    accuracy
                )
            }
        )

    # ========================================================
    # EVALUATE
    # ========================================================

    def evaluate(
        self,
        parameters,
        config
    ):

        print()
        print("=" * 60)
        print(">>> EVALUATION STARTED")
        print("=" * 60)

        # ====================================================
        # LOAD GLOBAL BACKBONE
        # ====================================================

        self.set_parameters(
            parameters
        )

        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        # ====================================================
        # VALIDATION
        # ====================================================

        with torch.no_grad():

            for (
                X_batch,
                y_batch
            ) in self.val_loader:

                X_batch = X_batch.to(
                    self.device
                )

                y_batch = y_batch.to(
                    self.device
                )

                outputs = self.model(
                    X_batch
                )

                loss = self.criterion(
                    outputs,
                    y_batch
                )

                batch_size = (
                    y_batch.size(0)
                )

                total_loss += (
                    loss.item()
                    * batch_size
                )

                predicted = torch.argmax(
                    outputs,
                    dim=1
                )

                correct += (
                    (
                        predicted == y_batch
                    )
                    .sum()
                    .item()
                )

                total += batch_size

        # ====================================================
        # METRICS
        # ====================================================

        avg_loss = (
            total_loss / total
            if total > 0
            else 0.0
        )

        accuracy = (
            correct / total
            if total > 0
            else 0.0
        )

        print()
        print(
            f"Validation Loss     : {avg_loss:.4f}"
        )

        print(
            f"Validation Accuracy : {accuracy:.4f}"
        )

        print(
            "Validation samples  :",
            total
        )

        return (
            float(avg_loss),
            total,
            {
                "accuracy": float(
                    accuracy
                )
            }
        )