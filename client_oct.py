import os
import gc
import random

from collections import defaultdict

import cv2
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from model import Net
from label_mapping import (
    map_oct_label,
    NUM_CLASSES,
    LABELS
)


# MEMORY / CPU SAFETY
cv2.setNumThreads(0)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
gc.collect()



# CONFIGURATION
SERVER_ADDRESS = "127.0.0.1:9090"
IMG_SIZE = 224
MAX_PER_CLASS = 500
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
FEDPROX_MU = 0.01
LOCAL_EPOCHS = 1
TRAIN_RATIO = 0.80
RANDOM_SEED = 42
DEVICE = torch.device("cpu")

# DATA PATH
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "classification",
    "RetinalOCT_Dataset",
    "train"
)


# OCT ORIGINAL CLASS NAMES
OCT_CLASS_NAMES = {
    0: "AMD",
    1: "CNV",
    2: "CSR",
    3: "DME",
    4: "DR",
    5: "DRUSEN",
    6: "MH"
}

# STARTUP
print("\n")
print("=" * 70)
print("STARTING OCT FEDERATED CLIENT")
print("=" * 70)

print("Server:", SERVER_ADDRESS)
print("Device:", DEVICE)
print("Data path:", DATA_PATH)
print("Data exists:", os.path.exists(DATA_PATH))
print("Batch size:", BATCH_SIZE)
print("Learning rate:", LEARNING_RATE)
print("FedProx mu:", FEDPROX_MU)
print("Local epochs:", LOCAL_EPOCHS)
print("Unified classes:", NUM_CLASSES)

print("\nGlobal label mapping:")

for index, label_name in enumerate(LABELS):
    print(f"  {index} -> {label_name}")

print("=" * 70)

#path check
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        "\nOCT dataset path not found:\n"
        + DATA_PATH
    )



#random seed
def set_seed(seed=RANDOM_SEED):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# image extensions
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


#load oct data directly
def load_oct_data():

    print("\n")
    print("=" * 70)
    print("SCANNING OCT DATASET")
    print("=" * 70)

    if not os.path.isdir(DATA_PATH):
        raise FileNotFoundError(
            f"OCT directory not found:\n{DATA_PATH}"
        )

    raw_data = []

    class_distribution = defaultdict(int)

    for folder_name in sorted(
        os.listdir(DATA_PATH)
    ):

        folder_path = os.path.join(
            DATA_PATH,
            folder_name
        )

        if not os.path.isdir(folder_path):
            continue

        normalized_name = (
            folder_name.strip()
            .lower()
        )

        # ----------------------------------------------------
        # FIND ORIGINAL OCT CLASS INDEX
        # ----------------------------------------------------

        original_label = None

        for index, class_name in OCT_CLASS_NAMES.items():

            if normalized_name == class_name.lower():

                original_label = index
                break

        # ----------------------------------------------------
        # UNKNOWN FOLDER
        # ----------------------------------------------------

        if original_label is None:

            print(
                f"WARNING: Unknown OCT folder skipped: "
                f"{folder_name}"
            )

            continue

        unified_label = map_oct_label(
            folder_name
        )

        # ----------------------------------------------------
        # UNSUPPORTED CLASS
        # ----------------------------------------------------

        if unified_label == -1:

            print(
                f"{folder_name:10s} -> EXCLUDED "
                f"(no verified unified mapping)"
            )

            continue

        # ----------------------------------------------------
        # COLLECT IMAGE PATHS
        # ----------------------------------------------------

        files = []

        for root, _, filenames in os.walk(
            folder_path
        ):

            for filename in filenames:

                extension = os.path.splitext(
                    filename
                )[1].lower()

                if extension not in IMAGE_EXTENSIONS:
                    continue

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

        # ----------------------------------------------------
        # SHUFFLE
        # ----------------------------------------------------

        random.shuffle(files)

        # ----------------------------------------------------
        # LIMIT CLASS
        # ----------------------------------------------------

        selected_files = files[
            :MAX_PER_CLASS
        ]

        print(
            f"{folder_name:10s} -> "
            f"{LABELS[unified_label]:22s} "
            f"| Available: {len(files):5d} "
            f"| Selected: {len(selected_files):5d}"
        )

        # ----------------------------------------------------
        # STORE
        # ----------------------------------------------------

        for image_path in selected_files:

            raw_data.append(
                (
                    image_path,
                    unified_label
                )
            )

            class_distribution[
                unified_label
            ] += 1

    # ========================================================
    # CHECK
    # ========================================================

    if len(raw_data) == 0:

        raise ValueError(
            "\nNo supported OCT images were found."
        )

    # ========================================================
    # FINAL DISTRIBUTION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL OCT UNIFIED DISTRIBUTION")
    print("=" * 70)

    for class_index in range(NUM_CLASSES):

        print(
            f"{class_index}: "
            f"{LABELS[class_index]:22s} -> "
            f"{class_distribution[class_index]} images"
        )

    print(
        "\nTotal usable OCT samples:",
        len(raw_data)
    )

    print("=" * 70)

    return raw_data



# OCT PYTORCH DATASET


class OCTDataset(Dataset):

    def __init__(self, data):

        self.data = data

        self.mean = torch.tensor(
            [
                0.485,
                0.456,
                0.406
            ],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        self.std = torch.tensor(
            [
                0.229,
                0.224,
                0.225
            ],
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(self.data)

    # ========================================================
    # GET ITEM
    # ========================================================

    def __getitem__(self, index):

        image_path, target = self.data[index]

        # ----------------------------------------------------
        # READ IMAGE
        # ----------------------------------------------------

        img = cv2.imread(
            image_path
        )

        if img is None:

            raise RuntimeError(
                "Could not read OCT image:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # GRAYSCALE -> BGR
        # ----------------------------------------------------

        if img.ndim == 2:

            img = cv2.cvtColor(
                img,
                cv2.COLOR_GRAY2BGR
            )

        # ----------------------------------------------------
        # BGRA -> BGR
        # ----------------------------------------------------

        elif (
            img.ndim == 3
            and img.shape[2] == 4
        ):

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGRA2BGR
            )

        # ----------------------------------------------------
        # CHECK CHANNELS
        # ----------------------------------------------------

        if (
            img.ndim != 3
            or img.shape[2] != 3
        ):

            raise RuntimeError(
                "Invalid OCT image shape:\n"
                f"{img.shape}\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # RESIZE
        # ----------------------------------------------------

        img = cv2.resize(
            img,
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        img = np.transpose(
            img,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # FLOAT32
        # ----------------------------------------------------

        img = np.ascontiguousarray(
            img,
            dtype=np.float32
        )

        # ----------------------------------------------------
        # [0,1]
        # ----------------------------------------------------

        img = torch.from_numpy(
            img
        )

        img = img / 255.0

        # ----------------------------------------------------
        # IMAGENET NORMALIZATION
        # ----------------------------------------------------

        img = (
            img - self.mean
        ) / self.std

        # ----------------------------------------------------
        # FINAL TARGET
        # ----------------------------------------------------

        target = int(target)

        if target < 0 or target >= NUM_CLASSES:

            raise ValueError(
                f"Invalid unified OCT target: {target}"
            )

        target = torch.tensor(
            target,
            dtype=torch.long
        )

        return (
            img,
            target
        )



# CREATE DATA LOADERS


def create_data_loaders():

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    data = load_oct_data()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    dataset = OCTDataset(
        data
    )

    if len(dataset) < 2:

        raise ValueError(
            "Not enough OCT samples for "
            "train/validation split."
        )

    # ========================================================
    # TRAIN / VALIDATION SPLIT
    # ========================================================

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
            "OCT training split is empty."
        )

    if val_size <= 0:

        raise RuntimeError(
            "OCT validation split is empty."
        )

    generator = (
        torch.Generator()
        .manual_seed(
            RANDOM_SEED
        )
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [
            train_size,
            val_size
        ],
        generator=generator
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # ========================================================
    # INFORMATION
    # ========================================================

    print("\n")
    print("=" * 70)
    print("OCT TRAIN / VALIDATION SPLIT")
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

    print(
        "Training batches  :",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print("=" * 70)

    return (
        train_loader,
        val_loader,
        train_size,
        val_size
    )



# OCT FLOWER CLIENT


class OCTClient(
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
        print("INITIALIZING OCT FLOWER CLIENT")
        print("=" * 70)

        self.device = DEVICE

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.train_size = train_size

        self.val_size = val_size

        # ====================================================
        # MODEL
        # ====================================================

        self.model = Net().to(
            self.device
        )

        print(
            "Device:",
            self.device
        )

        print(
            "Model:",
            "AlexNet 5-class Net"
        )

        print(
            "Output classes:",
            NUM_CLASSES
        )

        print("\nGlobal label mapping:")

        for index, label_name in enumerate(
            LABELS
        ):

            print(
                f"  {index} -> {label_name}"
            )

        # ====================================================
        # FEDERATED PARAMETER CHECK
        # ====================================================

        parameters = (
            self.model.get_shared_parameters()
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

        # ----------------------------------------------------
        # CURRENT MODEL MUST HAVE 10 TENSORS
        # ----------------------------------------------------

        if len(parameters) != 10:

            raise RuntimeError(
                "\nCurrent Net should expose exactly "
                "10 federated tensors.\n"
                f"Found: {len(parameters)}"
            )

        print("=" * 70)

        print(
            "Federated component:"
        )

        print(
            "Trainable AlexNet projection + "
            "5-class classification head"
        )

        print(
            "Frozen AlexNet convolutional layers"
        )

        print("=" * 70)

        print(
            "OCT CLIENT READY"
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
    # SET PARAMETERS
    # ========================================================

    def set_parameters(
        self,
        parameters
    ):

        if len(parameters) != 10:

            raise ValueError(
                "\nFederated parameter count mismatch.\n"
                f"Received: {len(parameters)}\n"
                "Expected: 10"
            )

        self.model.set_shared_parameters(
            parameters
        )

        print(
            "Received federated parameters:",
            len(parameters),
            "tensors"
        )

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
        print(">>> OCT FIT STARTED")
        print("=" * 70)

        # ----------------------------------------------------
        # RECEIVE GLOBAL MODEL
        # ----------------------------------------------------

        self.set_parameters(
            parameters
        )

        self.model.train()

        # ====================================================
        # GLOBAL SNAPSHOT FOR FEDPROX
        # ====================================================

        global_parameters = {}

        for name, parameter in (
            self.model.named_parameters()
        ):

            if not parameter.requires_grad:
                continue

            global_parameters[name] = (
                parameter.detach()
                .clone()
            )

        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer = optim.Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=LEARNING_RATE
        )

        # ====================================================
        # LOSS
        # ====================================================

        criterion = nn.CrossEntropyLoss()

        # ====================================================
        # LOCAL EPOCHS
        # ====================================================

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

        print(
            "Training samples:",
            self.train_size
        )

        total_loss = 0.0

        # ====================================================
        # LOCAL TRAINING
        # ====================================================

        for epoch in range(
            local_epochs
        ):

            epoch_loss = 0.0

            epoch_samples = 0

            correct = 0

            for images, targets in (
                self.train_loader
            ):

                images = images.to(
                    self.device
                )

                targets = targets.to(
                    self.device
                )

                # ------------------------------------------------
                # ZERO GRADIENT
                # ------------------------------------------------

                optimizer.zero_grad(
                    set_to_none=True
                )

                # ------------------------------------------------
                # FORWARD
                # ------------------------------------------------

                logits = self.model(
                    images
                )

                # ------------------------------------------------
                # OUTPUT CHECK
                # ------------------------------------------------

                if logits.ndim != 2:

                    raise RuntimeError(
                        "Unexpected model output shape: "
                        f"{tuple(logits.shape)}"
                    )

                if logits.shape[1] != NUM_CLASSES:

                    raise RuntimeError(
                        "Model output class count mismatch.\n"
                        f"Model output: {logits.shape[1]}\n"
                        f"Expected: {NUM_CLASSES}"
                    )
                    
                # CROSS ENTROPY
                classification_loss = criterion(
                    logits,
                    targets
                )
                # FEDPROX
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
                # TOTAL LOSS


                loss = (
                    classification_loss
                    + proximal_loss
                )

                # ------------------------------------------------
                # BACKPROP
                # ------------------------------------------------

                loss.backward()

                optimizer.step()

                # ------------------------------------------------
                # STATISTICS
                # ------------------------------------------------

                batch_size = images.size(0)

                epoch_loss += (
                    loss.item()
                    * batch_size
                )

                epoch_samples += batch_size

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                correct += (
                    predictions == targets
                ).sum().item()

                # ------------------------------------------------
                # MEMORY CLEANUP
                # ------------------------------------------------

                del images
                del targets
                del logits
                del predictions
                del loss

            # ----------------------------------------------------
            # AVERAGE LOSS
            # ----------------------------------------------------

            epoch_loss /= max(
                epoch_samples,
                1
            )

            epoch_accuracy = (
                correct
                /
                max(
                    epoch_samples,
                    1
                )
            )

            total_loss = epoch_loss

            print(
                f"Epoch "
                f"{epoch + 1}/"
                f"{local_epochs}"
                f" | Loss: "
                f"{epoch_loss:.4f}"
                f" | Accuracy: "
                f"{epoch_accuracy:.4f}"
            )

            gc.collect()

        # ====================================================
        # UPDATED PARAMETERS
        # ====================================================

        updated_parameters = (
            self.model.get_shared_parameters()
        )

        if len(updated_parameters) != 10:

            raise RuntimeError(
                "\nUpdated federated parameter count "
                "must be 10.\n"
                f"Found: {len(updated_parameters)}"
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
            "Final local loss:",
            f"{total_loss:.4f}"
        )

        print("=" * 70)
        print(">>> OCT FIT FINISHED")
        print("=" * 70)

        del optimizer
        del criterion

        gc.collect()

        return (
            updated_parameters,
            self.train_size,
            {
                "loss": float(
                    total_loss
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
        print(">>> OCT EVALUATION")
        print("=" * 70)

        # ----------------------------------------------------
        # RECEIVE PARAMETERS
        # ----------------------------------------------------

        self.set_parameters(
            parameters
        )

        self.model.eval()

        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0

        total_samples = 0

        correct = 0

        # ====================================================
        # VALIDATION
        # ====================================================

        with torch.no_grad():

            for images, targets in (
                self.val_loader
            ):

                images = images.to(
                    self.device
                )

                targets = targets.to(
                    self.device
                )

                # ------------------------------------------------
                # FORWARD
                # ------------------------------------------------

                logits = self.model(
                    images
                )

                # ------------------------------------------------
                # LOSS
                # ------------------------------------------------

                loss = criterion(
                    logits,
                    targets
                )

                # ------------------------------------------------
                # STATISTICS
                # ------------------------------------------------

                batch_size = images.size(0)

                total_loss += (
                    loss.item()
                    * batch_size
                )

                total_samples += batch_size

                predictions = torch.argmax(
                    logits,
                    dim=1
                )

                correct += (
                    predictions == targets
                ).sum().item()

        # ====================================================
        # AVERAGE
        # ====================================================

        average_loss = (
            total_loss
            /
            max(
                total_samples,
                1
            )
        )

        accuracy = (
            correct
            /
            max(
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

        del criterion

        gc.collect()

        return (
            float(average_loss),
            self.val_size,
            {
                "loss": float(
                    average_loss
                ),
                "accuracy": float(
                    accuracy
                )
            }
        )



# MAIN


def main():

    print("\n")
    print("=" * 70)
    print("OCT FLOWER CLIENT")
    print("=" * 70)

    set_seed()

    # ========================================================
    # DATA
    # ========================================================

    (
        train_loader,
        val_loader,
        train_size,
        val_size
    ) = create_data_loaders()

    # ========================================================
    # CLIENT
    # ========================================================

    client = OCTClient(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=train_size,
        val_size=val_size
    )

    # ========================================================
    # PARAMETER COUNT
    # ========================================================

    parameter_count = len(
        client.model.get_shared_parameters()
    )

    if parameter_count != 10:

        raise RuntimeError(
            "\nOCT client has incorrect federated "
            "parameter count.\n"
            f"Found: {parameter_count}\n"
            "Expected: 10"
        )

    print("\n")
    print("=" * 70)
    print("OCT FLOWER CLIENT")
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

    print(
        "Output classes:",
        NUM_CLASSES
    )

    print(
        "10-round setup:",
        "Controlled by server"
    )

    print("=" * 70)

    # ========================================================
    # CONNECT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("CONNECTING OCT CLIENT TO FLOWER SERVER")
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



# ENTRY POINT


if __name__ == "__main__":

    main()