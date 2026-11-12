import os
import random
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import flwr as fl

from model import Net
from label_mapping import (
    map_odir_label,
    NUM_CLASSES,
    LABELS
)


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_NAME = "ODIR-5K"

SERVER_ADDRESS = "127.0.0.1:9090"

IMG_SIZE = 224
BATCH_SIZE = 8

LOCAL_EPOCHS = 2
LEARNING_RATE = 1e-4

MAX_SAMPLES_PER_CLASS = 300

VAL_SPLIT = 0.20

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# RANDOM SEED
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "classification",
    "ODIR-5K_Final"
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)


# ============================================================
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
}


# ============================================================
# DATASET
# ============================================================

class ODIRDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples
        self.transform = transform

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        image_path, label = self.samples[index]

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

        except Exception as e:

            print("\nError reading image:")
            print(image_path)
            print(e)

            image = Image.new(
                "RGB",
                (
                    IMG_SIZE,
                    IMG_SIZE
                )
            )

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# TRAIN TRANSFORM
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=10
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


# ============================================================
# VALIDATION TRANSFORM
# ============================================================

val_transform = transforms.Compose([

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


# ============================================================
# FIND ODIR IMAGES
# ============================================================

def find_images():

    samples = []

    if not os.path.exists(TRAIN_DIR):

        raise FileNotFoundError(
            "\nODIR-5K training directory not found:\n"
            f"{TRAIN_DIR}"
        )

    print("\nScanning ODIR-5K folders...")

    for root, dirs, files in os.walk(TRAIN_DIR):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            image_path = os.path.join(
                root,
                filename
            )

            # ------------------------------------------------
            # The folder immediately containing the image
            # is used as the ODIR label.
            # ------------------------------------------------

            folder_name = os.path.basename(
                root
            )

            unified_label = map_odir_label(
                folder_name
            )

            # ------------------------------------------------
            # Ignore unsupported ODIR labels:
            #
            # hypertension
            # myopia
            # ------------------------------------------------

            if unified_label == -1:
                continue

            samples.append(
                (
                    image_path,
                    unified_label
                )
            )

    return samples


# ============================================================
# BALANCE / LIMIT DATA
# ============================================================

def balance_samples(samples):

    grouped = {
        class_id: []
        for class_id in range(NUM_CLASSES)
    }

    for image_path, label in samples:

        if label in grouped:

            grouped[label].append(
                (
                    image_path,
                    label
                )
            )

    print("\nODIR unified classes before limiting:")
    print("-" * 60)

    for class_id in range(NUM_CLASSES):

        print(
            f"{class_id}: "
            f"{LABELS[class_id]:25s} "
            f"{len(grouped[class_id])} images"
        )

    balanced = []

    for class_id in range(NUM_CLASSES):

        class_samples = grouped[class_id]

        random.shuffle(
            class_samples
        )

        if MAX_SAMPLES_PER_CLASS is not None:

            class_samples = class_samples[
                :MAX_SAMPLES_PER_CLASS
            ]

        balanced.extend(
            class_samples
        )

    random.shuffle(
        balanced
    )

    return balanced


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n")
    print("=" * 65)
    print("                 ODIR-5K DATA LOADING")
    print("=" * 65)

    print("\nDATA DIRECTORY:")
    print(DATA_DIR)

    print("\nTRAIN DIRECTORY:")
    print(TRAIN_DIR)

    samples = find_images()

    print(
        f"\nUsable ODIR images discovered: "
        f"{len(samples)}"
    )

    if len(samples) == 0:

        raise RuntimeError(
            "\nNo supported ODIR images found.\n"
            "Check the ODIR-5K folder structure."
        )

    # --------------------------------------------------------
    # BALANCE DATA
    # --------------------------------------------------------

    samples = balance_samples(
        samples
    )

    # --------------------------------------------------------
    # FINAL DISTRIBUTION
    # --------------------------------------------------------

    distribution = {
        class_id: 0
        for class_id in range(NUM_CLASSES)
    }

    for _, label in samples:

        distribution[label] += 1

    print("\n")
    print("=" * 65)
    print("          FINAL ODIR UNIFIED DISTRIBUTION")
    print("=" * 65)

    for class_id in range(NUM_CLASSES):

        print(
            f"{class_id}: "
            f"{LABELS[class_id]:25s} -> "
            f"{distribution[class_id]} images"
        )

    print(
        f"\nTotal usable ODIR samples: "
        f"{len(samples)}"
    )

    # --------------------------------------------------------
    # TRAIN / VALIDATION SPLIT
    # --------------------------------------------------------

    random.shuffle(
        samples
    )

    split_index = int(
        len(samples)
        * (1 - VAL_SPLIT)
    )

    train_samples = samples[
        :split_index
    ]

    val_samples = samples[
        split_index:
    ]

    train_dataset = ODIRDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = ODIRDataset(
        val_samples,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    print("\nTraining samples  :", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    print(
        "Training batches  :",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(val_loader)
    )

    print("=" * 65)

    return (
        train_loader,
        val_loader
    )


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    model,
    train_loader,
    optimizer,
    criterion
):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += (
            loss.item()
            * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    average_loss = (
        total_loss / total
        if total > 0
        else 0.0
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    val_loader,
    criterion
):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            outputs = model(
                images
            )

            loss = criterion(
                outputs,
                labels
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    average_loss = (
        total_loss / total
        if total > 0
        else 0.0
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return (
        average_loss,
        accuracy
    )


# ============================================================
# FLOWER CLIENT
# ============================================================

class ODIRClient(
    fl.client.NumPyClient
):

    def __init__(self):

        print("\n")
        print("=" * 65)
        print("             INITIALIZING ODIR CLIENT")
        print("=" * 65)

        print(
            f"Device: {DEVICE}"
        )

        print(
            f"Unified classes: {NUM_CLASSES}"
        )

        print("\nGlobal label mapping:")

        for class_id, class_name in enumerate(LABELS):

            print(
                f"  {class_id} -> {class_name}"
            )

        # ----------------------------------------------------
        # CURRENT FEDERATED MODEL
        # ----------------------------------------------------

        self.model = (
            Net()
            .to(DEVICE)
        )

        # ----------------------------------------------------
        # LOAD ODIR DATA
        # ----------------------------------------------------

        (
            self.train_loader,
            self.val_loader
        ) = load_data()

        # ----------------------------------------------------
        # LOSS
        # ----------------------------------------------------

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        # ----------------------------------------------------
        # OPTIMIZER
        # ----------------------------------------------------

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        # ----------------------------------------------------
        # VERIFY FEDERATED PARAMETERS
        # ----------------------------------------------------

        federated_parameters = (
            self.model.get_shared_parameters()
        )

        print(
            f"\nFederated tensors: "
            f"{len(federated_parameters)}"
        )

        expected_tensors = 10

        if len(federated_parameters) != expected_tensors:

            raise RuntimeError(
                "\nFederated parameter count mismatch!\n"
                f"Expected: {expected_tensors}\n"
                f"Current : {len(federated_parameters)}\n"
                "Check model.py."
            )

        print("\nFederated parameter shapes:")

        for index, parameter in enumerate(
            federated_parameters
        ):

            print(
                f"Tensor {index}: "
                f"{parameter.shape}"
            )

        print("=" * 65)


    # ========================================================
    # GET PARAMETERS
    # ========================================================

    def get_parameters(
        self,
        config
    ):

        return (
            self.model.get_shared_parameters()
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
        print("=" * 65)
        print("                ODIR LOCAL TRAINING")
        print("=" * 65)

        # ----------------------------------------------------
        # Verify received parameter count
        # ----------------------------------------------------

        expected = len(
            self.model.get_shared_parameters()
        )

        received = len(
            parameters
        )

        if received != expected:

            raise RuntimeError(
                "\nFederated parameter count mismatch!\n"
                f"Server sent: {received}\n"
                f"Model expects: {expected}\n"
            )

        # ----------------------------------------------------
        # Load global model parameters
        # ----------------------------------------------------

        self.model.set_shared_parameters(
            parameters
        )

        # ----------------------------------------------------
        # Local epochs
        # ----------------------------------------------------

        epochs = int(
            config.get(
                "local_epochs",
                LOCAL_EPOCHS
            )
        )

        print(
            f"Local epochs: {epochs}"
        )

        last_loss = 0.0
        last_accuracy = 0.0

        # ----------------------------------------------------
        # Local training
        # ----------------------------------------------------

        for epoch in range(epochs):

            loss, accuracy = train_model(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion
            )

            last_loss = loss
            last_accuracy = accuracy

            print(
                f"Epoch "
                f"{epoch + 1}/{epochs} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

        print("=" * 65)

        # ----------------------------------------------------
        # Return updated shared parameters
        # ----------------------------------------------------

        return (

            self.model.get_shared_parameters(),

            len(
                self.train_loader.dataset
            ),

            {
                "train_loss": float(
                    last_loss
                ),

                "train_accuracy": float(
                    last_accuracy
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

        # ----------------------------------------------------
        # Verify parameter count
        # ----------------------------------------------------

        expected = len(
            self.model.get_shared_parameters()
        )

        received = len(
            parameters
        )

        if received != expected:

            raise RuntimeError(
                "\nFederated parameter count mismatch!\n"
                f"Server sent: {received}\n"
                f"Model expects: {expected}\n"
            )

        # ----------------------------------------------------
        # Load global parameters
        # ----------------------------------------------------

        self.model.set_shared_parameters(
            parameters
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        loss, accuracy = evaluate_model(
            self.model,
            self.val_loader,
            self.criterion
        )

        print(
            f"\nODIR Validation Loss: "
            f"{loss:.4f}"
        )

        print(
            f"ODIR Validation Accuracy: "
            f"{accuracy:.4f}"
        )

        return (

            float(loss),

            len(
                self.val_loader.dataset
            ),

            {
                "val_accuracy": float(
                    accuracy
                )
            }
        )


# ============================================================
# START CLIENT
# ============================================================

if __name__ == "__main__":

    client = ODIRClient()

    print("\n")
    print("=" * 65)
    print("          STARTING ODIR-5K FLOWER CLIENT")
    print("=" * 65)

    print(
        f"Server: {SERVER_ADDRESS}"
    )

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Classes: {NUM_CLASSES}"
    )

    print(
        "Federated tensors:",
        len(
            client.model.get_shared_parameters()
        )
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Modern Flower API
    # --------------------------------------------------------

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client()
    )