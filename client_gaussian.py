import os
import random
import numpy as np
from PIL import Image

import flwr as fl

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split

from torchvision import transforms

from model import Net
from label_mapping import (
    map_gaussian_label,
    NUM_CLASSES,
    LABELS
)


# ============================================================
# CONFIGURATION
# ============================================================

SERVER_ADDRESS = "127.0.0.1:9090"

IMG_SIZE = 224

BATCH_SIZE = 16
LEARNING_RATE = 1e-4

FEDPROX_MU = 0.01

TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# Maximum number of images taken from each Gaussian folder.
# None = use all images.
MAX_SAMPLES_PER_CLASS = 1050

# Local epochs per federated round
LOCAL_EPOCHS = 1


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# DATA PATH
# ============================================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "classification",
    "gaussian_filtered_images"
)


# ============================================================
# SEED
# ============================================================

def set_seed(seed=RANDOM_SEED):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# START INFORMATION
# ============================================================

print("\n")
print("=" * 70)
print("STARTING GAUSSIAN FEDERATED CLIENT")
print("=" * 70)

print("Server       :", SERVER_ADDRESS)
print("Device       :", DEVICE)
print("Data path    :", DATA_PATH)
print("Data exists  :", os.path.exists(DATA_PATH))
print("Local epochs :", LOCAL_EPOCHS)
print("Batch size   :", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("FedProx mu   :", FEDPROX_MU)
print("Output classes:", NUM_CLASSES)

print("\nGlobal label mapping:")

for index, label_name in enumerate(LABELS):
    print(f"  {index} -> {label_name}")

print("=" * 70)


# ============================================================
# DATA PATH CHECK
# ============================================================

if not os.path.exists(DATA_PATH):

    raise FileNotFoundError(
        "\nGaussian dataset path not found:\n"
        + DATA_PATH
    )


# ============================================================
# GAUSSIAN DATASET
# ============================================================

class GaussianDataset(Dataset):

    def __init__(
        self,
        data_path,
        max_samples_per_class=None
    ):

        self.samples = []

        self.transform = transforms.Compose([
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE)
            ),

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406
                ],
                std=[
                    0.229,
                    0.224,
                    0.225
                ]
            )
        ])

        print("\n")
        print("=" * 70)
        print("LOADING GAUSSIAN DATASET")
        print("=" * 70)

        # ----------------------------------------------------
        # Expected Gaussian folders
        # ----------------------------------------------------

        expected_folders = [
            "No_DR",
            "Mild",
            "Moderate",
            "Severe",
            "Proliferate_DR"
        ]

        # ----------------------------------------------------
        # Scan folders
        # ----------------------------------------------------

        for folder_name in expected_folders:

            folder_path = os.path.join(
                data_path,
                folder_name
            )

            if not os.path.isdir(folder_path):

                print(
                    f"WARNING: Folder not found: "
                    f"{folder_name}"
                )

                continue

            # ------------------------------------------------
            # Global 5-class mapping
            # ------------------------------------------------

            label = map_gaussian_label(
                folder_name
            )

            if label == -1:

                print(
                    f"Skipping unsupported folder: "
                    f"{folder_name}"
                )

                continue

            # ------------------------------------------------
            # Find images
            # ------------------------------------------------

            image_files = []

            for filename in os.listdir(
                folder_path
            ):

                if filename.lower().endswith(
                    (
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".bmp",
                        ".tif",
                        ".tiff"
                    )
                ):

                    image_files.append(
                        filename
                    )

            # ------------------------------------------------
            # Shuffle
            # ------------------------------------------------

            random.shuffle(
                image_files
            )

            # ------------------------------------------------
            # Limit samples
            # ------------------------------------------------

            if (
                max_samples_per_class
                is not None
            ):

                image_files = image_files[
                    :max_samples_per_class
                ]

            # ------------------------------------------------
            # Add samples
            # ------------------------------------------------

            for filename in image_files:

                image_path = os.path.join(
                    folder_path,
                    filename
                )

                self.samples.append(
                    (
                        image_path,
                        label
                    )
                )

            print(
                f"{folder_name:20s} -> "
                f"{label} ({LABELS[label]}) : "
                f"{len(image_files)} images"
            )

        # ----------------------------------------------------
        # Dataset check
        # ----------------------------------------------------

        if len(self.samples) == 0:

            raise RuntimeError(
                "No valid Gaussian images found."
            )

        print("\nTotal Gaussian samples:")
        print(len(self.samples))

        print("=" * 70)

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.samples
        )

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            image = self.transform(
                image
            )

            label = torch.tensor(
                label,
                dtype=torch.long
            )

            return (
                image,
                label
            )

        except Exception as error:

            print(
                "\nWARNING: Failed to load image:"
            )

            print(
                image_path
            )

            print(
                error
            )

            # Try another sample
            new_index = (
                index + 1
            ) % len(self.samples)

            return self.__getitem__(
                new_index
            )


# ============================================================
# CREATE DATA LOADERS
# ============================================================

def create_data_loaders():

    dataset = GaussianDataset(
        DATA_PATH,
        MAX_SAMPLES_PER_CLASS
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    train_size = int(
        TRAIN_RATIO
        * len(dataset)
    )

    val_size = (
        len(dataset)
        - train_size
    )

    if train_size <= 0:

        raise RuntimeError(
            "Training dataset is empty."
        )

    if val_size <= 0:

        raise RuntimeError(
            "Validation dataset is empty."
        )

    generator = (
        torch.Generator()
        .manual_seed(
            RANDOM_SEED
        )
    )

    (
        train_dataset,
        val_dataset
    ) = random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=generator
    )

    # --------------------------------------------------------
    # TRAIN LOADER
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # --------------------------------------------------------
    # VALIDATION LOADER
    # --------------------------------------------------------

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GAUSSIAN TRAIN / VALIDATION SPLIT")
    print("=" * 70)

    print(
        "Training samples  :",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    print(
        "Total samples     :",
        len(dataset)
    )

    print("=" * 70)

    return (
        train_loader,
        val_loader,
        len(train_dataset),
        len(val_dataset)
    )


# ============================================================
# GAUSSIAN FLOWER CLIENT
# ============================================================

class GaussianClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        train_loader,
        val_loader,
        train_size,
        val_size
    ):

        print("\n")
        print("=" * 70)
        print("INITIALIZING GAUSSIAN CLIENT")
        print("=" * 70)

        self.device = DEVICE

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.train_size = train_size
        self.val_size = val_size

        # ----------------------------------------------------
        # CURRENT 5-CLASS MODEL
        # ----------------------------------------------------

        self.model = Net().to(
            self.device
        )

        # ----------------------------------------------------
        # PARAMETER CHECK
        # ----------------------------------------------------

        parameters = (
            self.model.get_shared_parameters()
        )

        print(
            "Device:",
            self.device
        )

        print(
            "Model: Current Net / AlexNet"
        )

        print(
            "Output classes:",
            NUM_CLASSES
        )

        print("\nGlobal classes:")

        for index, name in enumerate(
            LABELS
        ):

            print(
                f"  {index} -> {name}"
            )

        print("\n")
        print("=" * 70)
        print("FEDERATED PARAMETER CHECK")
        print("=" * 70)

        print(
            "Federated tensors:",
            len(parameters)
        )

        for index, parameter in enumerate(
            parameters
        ):

            print(
                f"  Tensor {index}: "
                f"shape={parameter.shape}"
            )

        print("=" * 70)

        # ----------------------------------------------------
        # IMPORTANT
        # ----------------------------------------------------

        if len(parameters) != 10:

            raise RuntimeError(
                "\nCURRENT MODEL DOES NOT HAVE "
                "10 FEDERATED TENSORS.\n"
                f"Found: {len(parameters)}\n"
                "Expected: 10\n"
                "Check model.py."
            )

        print(
            "GAUSSIAN CLIENT READY"
        )

        print("=" * 70)

    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config
    ):

        parameters = (
            self.model.get_shared_parameters()
        )

        print(
            "\nSending federated parameters:",
            len(parameters),
            "tensors"
        )

        return parameters

    # ========================================================
    # FIT
    # ========================================================

    def fit(
        self,
        parameters,
        config
    ):

        print("\n")
        print("=" * 70)
        print(">>> GAUSSIAN FIT STARTED")
        print("=" * 70)

        # ----------------------------------------------------
        # RECEIVE GLOBAL MODEL
        # ----------------------------------------------------

        self.model.set_shared_parameters(
            parameters
        )

        print(
            "Received federated tensors:",
            len(parameters)
        )

        # ----------------------------------------------------
        # FEDPROX GLOBAL SNAPSHOT
        # ----------------------------------------------------

        global_parameters = {}

        for name, parameter in (
            self.model.named_parameters()
        ):

            if parameter.requires_grad:

                global_parameters[name] = (
                    parameter.detach()
                    .clone()
                )

        # ----------------------------------------------------
        # MODEL TRAIN MODE
        # ----------------------------------------------------

        self.model.train()

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        criterion = nn.CrossEntropyLoss()

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        optimizer = optim.Adam(
            [
                parameter
                for parameter in
                self.model.parameters()
                if parameter.requires_grad
            ],
            lr=LEARNING_RATE
        )

        # ----------------------------------------------------
        # LOCAL EPOCHS
        # ----------------------------------------------------

        local_epochs = int(
            config.get(
                "local_epochs",
                LOCAL_EPOCHS
            )
        )

        print(
            "Local epochs:",
            local_epochs
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # ====================================================
        # TRAINING
        # ====================================================

        for epoch in range(
            local_epochs
        ):

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_samples = 0

            for (
                images,
                labels
            ) in self.train_loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                # ------------------------------------------------
                # ZERO GRADIENT
                # ------------------------------------------------

                optimizer.zero_grad()

                # ------------------------------------------------
                # FORWARD
                # ------------------------------------------------

                logits = self.model(
                    images
                )

                # ------------------------------------------------
                # CROSS ENTROPY
                # ------------------------------------------------

                classification_loss = (
                    criterion(
                        logits,
                        labels
                    )
                )

                # ------------------------------------------------
                # FEDPROX
                # ------------------------------------------------

                proximal_term = torch.tensor(
                    0.0,
                    device=self.device
                )

                for name, parameter in (
                    self.model.named_parameters()
                ):

                    if not parameter.requires_grad:
                        continue

                    proximal_term += torch.sum(
                        (
                            parameter
                            - global_parameters[name]
                        ) ** 2
                    )

                proximal_loss = (
                    FEDPROX_MU
                    / 2.0
                ) * proximal_term

                # ------------------------------------------------
                # TOTAL LOSS
                # ------------------------------------------------

                loss = (
                    classification_loss
                    + proximal_loss
                )

                # ------------------------------------------------
                # BACKPROPAGATION
                # ------------------------------------------------

                loss.backward()

                optimizer.step()

                # ------------------------------------------------
                # STATISTICS
                # ------------------------------------------------

                batch_size = (
                    images.size(0)
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                correct = (
                    predictions == labels
                ).sum().item()

                epoch_loss += (
                    loss.item()
                    * batch_size
                )

                epoch_correct += correct

                epoch_samples += (
                    batch_size
                )

            # ----------------------------------------------------
            # EPOCH METRICS
            # ----------------------------------------------------

            epoch_loss /= max(
                epoch_samples,
                1
            )

            epoch_accuracy = (
                epoch_correct
                / max(
                    epoch_samples,
                    1
                )
            )

            total_loss = epoch_loss
            total_correct = epoch_correct
            total_samples = epoch_samples

            print(
                f"Epoch {epoch + 1}/{local_epochs} | "
                f"Loss: {epoch_loss:.4f} | "
                f"Accuracy: {epoch_accuracy:.4f}"
            )

        # ====================================================
        # UPDATED PARAMETERS
        # ====================================================

        updated_parameters = (
            self.model.get_shared_parameters()
        )

        print("\n")
        print(
            "Returning federated tensors:",
            len(updated_parameters)
        )

        print(
            "Training samples:",
            self.train_size
        )

        print(
            "Training loss:",
            f"{total_loss:.4f}"
        )

        training_accuracy = (
            total_correct
            / max(
                total_samples,
                1
            )
        )

        print(
            "Training accuracy:",
            f"{training_accuracy:.4f}"
        )

        print("=" * 70)
        print(">>> GAUSSIAN FIT FINISHED")
        print("=" * 70)

        return (
            updated_parameters,
            self.train_size,
            {
                "loss": float(
                    total_loss
                ),
                "accuracy": float(
                    training_accuracy
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

        print("\n")
        print("=" * 70)
        print(">>> GAUSSIAN EVALUATION")
        print("=" * 70)

        # ----------------------------------------------------
        # RECEIVE GLOBAL PARAMETERS
        # ----------------------------------------------------

        self.model.set_shared_parameters(
            parameters
        )

        self.model.eval()

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        with torch.no_grad():

            for (
                images,
                labels
            ) in self.val_loader:

                images = images.to(
                    self.device
                )

                labels = labels.to(
                    self.device
                )

                logits = self.model(
                    images
                )

                loss = criterion(
                    logits,
                    labels
                )

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                batch_size = (
                    images.size(0)
                )

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_correct += (
                    predictions == labels
                ).sum().item()

                total_samples += (
                    batch_size
                )

        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        average_loss = (
            total_loss
            / max(
                total_samples,
                1
            )
        )

        accuracy = (
            total_correct
            / max(
                total_samples,
                1
            )
        )

        print(
            "Validation loss:",
            f"{average_loss:.4f}"
        )

        print(
            "Validation accuracy:",
            f"{accuracy:.4f}"
        )

        print(
            "Validation samples:",
            self.val_size
        )

        print("=" * 70)

        return (
            float(average_loss),
            self.val_size,
            {
                "accuracy": float(
                    accuracy
                )
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("STARTING GAUSSIAN FEDERATED CLIENT")
    print("=" * 70)

    print(
        "Server:",
        SERVER_ADDRESS
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Output classes:",
        NUM_CLASSES
    )

    print(
        "Local epochs:",
        LOCAL_EPOCHS
    )

    print("=" * 70)

    # --------------------------------------------------------
    # SEED
    # --------------------------------------------------------

    set_seed()

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        train_size,
        val_size
    ) = create_data_loaders()

    # --------------------------------------------------------
    # CLIENT
    # --------------------------------------------------------

    client = GaussianClient(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=train_size,
        val_size=val_size
    )

    # --------------------------------------------------------
    # FINAL PARAMETER CHECK
    # --------------------------------------------------------

    parameter_count = len(
        client.model.get_shared_parameters()
    )

    print("\n")
    print("=" * 70)
    print("GAUSSIAN FLOWER CLIENT")
    print("=" * 70)

    print(
        "Server:",
        SERVER_ADDRESS
    )

    print(
        "Federated tensors:",
        parameter_count
    )

    print(
        "Training samples:",
        train_size
    )

    print(
        "Validation samples:",
        val_size
    )

    print("=" * 70)

    if parameter_count != 10:

        raise RuntimeError(
            "\nSTOPPING.\n"
            f"Current model has {parameter_count} "
            "federated tensors.\n"
            "Expected exactly 10."
        )

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CONNECTING GAUSSIAN CLIENT TO FLOWER SERVER")
    print("=" * 70)

    print(
        "Server:",
        SERVER_ADDRESS
    )

    print(
        "Waiting for federated rounds..."
    )

    print("=" * 70)

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()