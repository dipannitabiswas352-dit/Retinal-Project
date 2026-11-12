import os
import gc
import random

import cv2
import flwr as fl
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from model import Net
from label_mapping import (
    NUM_CLASSES,
    LABELS
)


# ============================================================
# CPU / MEMORY SAFETY
# ============================================================

cv2.setNumThreads(0)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

gc.collect()


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_NAME = "RFMiD"

SERVER_ADDRESS = "127.0.0.1:9090"

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "classification",
    "RFMiD_dataset",
    "Training_set"
)

CSV_PATH = os.path.join(
    DATA_DIR,
    "RFMiD_Training_Labels.csv"
)

IMG_SIZE = 224

BATCH_SIZE = 2

LEARNING_RATE = 1e-4

LOCAL_EPOCHS = 1

MAX_SAMPLES_PER_CLASS = 200

TRAIN_RATIO = 0.80

RANDOM_SEED = 42

DEVICE = torch.device("cpu")


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32
)

STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32
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
# CLASS INFORMATION
# ============================================================

CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "AMD",
    "Cataract",
    "Normal"
]


# RFMiD supported class IDs
RFMiD_DR = 0
RFMiD_GLAUCOMA = 1
RFMiD_AMD = 2
RFMiD_CATARACT = 3
RFMiD_NORMAL = 4


# ============================================================
# STARTUP
# ============================================================

print("\n")
print("=" * 70)
print("STARTING RFMID FEDERATED CLIENT")
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
    "Project root:",
    PROJECT_ROOT
)

print(
    "Data directory:",
    DATA_DIR
)

print(
    "CSV:",
    CSV_PATH
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print(
    "Local epochs:",
    LOCAL_EPOCHS
)

print(
    "Maximum samples per class:",
    MAX_SAMPLES_PER_CLASS
)

print(
    "Unified classes:",
    NUM_CLASSES
)

print("\nGlobal label mapping:")

for index, label in enumerate(LABELS):

    print(
        f"  {index} -> {label}"
    )

print("=" * 70)


# ============================================================
# PATH CHECK
# ============================================================

if not os.path.exists(DATA_DIR):

    raise FileNotFoundError(
        "\nRFMiD directory not found:\n"
        f"{DATA_DIR}"
    )


if not os.path.exists(CSV_PATH):

    raise FileNotFoundError(
        "\nRFMiD CSV not found:\n"
        f"{CSV_PATH}"
    )


# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(
    seed=RANDOM_SEED
):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# NORMALIZE COLUMN NAME
# ============================================================

def normalize_column_name(
    name
):

    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
    )


# ============================================================
# FIND COLUMN
# ============================================================

def find_column(
    columns,
    aliases
):

    normalized_columns = {
        normalize_column_name(column): column
        for column in columns
    }

    for alias in aliases:

        key = normalize_column_name(
            alias
        )

        if key in normalized_columns:

            return normalized_columns[key]

    return None


# ============================================================
# FIND IMAGE COLUMN
# ============================================================

def find_image_column(
    columns
):

    aliases = [
        "ID",
        "Image",
        "ImageName",
        "ImageID",
        "filename",
        "FileName",
        "file",
        "name"
    ]

    result = find_column(
        columns,
        aliases
    )

    if result is not None:

        return result

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    for column in columns:

        normalized = normalize_column_name(
            column
        )

        if (
            "image" in normalized
            or
            "filename" in normalized
        ):

            return column

    return None


# ============================================================
# FIND RFMiD DISEASE COLUMNS
# ============================================================

def find_disease_columns(
    columns
):

    dr_column = find_column(
        columns,
        [
            "DR",
            "DiabeticRetinopathy",
            "Diabetic_Retinopathy",
            "Retinopathy"
        ]
    )

    glaucoma_column = find_column(
        columns,
        [
            "G",
            "Glaucoma"
        ]
    )

    amd_column = find_column(
        columns,
        [
            "ARMD",
            "AMD",
            "AgeRelatedMacularDegeneration",
            "Age_Related_Macular_Degeneration"
        ]
    )

    cataract_column = find_column(
        columns,
        [
            "C",
            "Cataract"
        ]
    )

    disease_risk_column = find_column(
        columns,
        [
            "Disease_Risk",
            "DiseaseRisk"
        ]
    )

    return {
        "dr": dr_column,
        "glaucoma": glaucoma_column,
        "amd": amd_column,
        "cataract": cataract_column,
        "disease_risk": disease_risk_column
    }


# ============================================================
# CONVERT VALUE TO BINARY
# ============================================================

def to_binary(
    value
):

    if pd.isna(value):

        return 0

    # --------------------------------------------------------
    # Numeric
    # --------------------------------------------------------

    try:

        number = float(value)

        return 1 if number == 1 else 0

    except Exception:

        pass

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text = (
        str(value)
        .strip()
        .lower()
    )

    if text in {
        "1",
        "yes",
        "true",
        "present",
        "positive",
        "disease"
    }:

        return 1

    return 0


# ============================================================
# NORMALIZE IMAGE NAME
# ============================================================

def normalize_image_name(
    value
):

    value = str(
        value
    ).strip()

    value = value.replace(
        "\\",
        "/"
    )

    value = os.path.basename(
        value
    )

    return os.path.splitext(
        value
    )[0].lower()


# ============================================================
# BUILD IMAGE LOOKUP
# ============================================================

def build_image_lookup():

    print("\n")
    print("=" * 70)
    print("BUILDING RFMID IMAGE LOOKUP")
    print("=" * 70)

    lookup = {}

    for filename in os.listdir(
        DATA_DIR
    ):

        full_path = os.path.join(
            DATA_DIR,
            filename
        )

        if not os.path.isfile(
            full_path
        ):

            continue

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension not in IMAGE_EXTENSIONS:

            continue

        stem = os.path.splitext(
            filename
        )[0].lower()

        lookup[stem] = full_path

    print(
        "Images found:",
        len(lookup)
    )

    return lookup


# ============================================================
# CREATE SINGLE RFMiD LABEL
# ============================================================

def create_rfid_label(
    dr_value,
    glaucoma_value,
    amd_value,
    cataract_value,
    disease_risk_value
):
    """
    Convert RFMiD multi-label information into one
    unified 5-class label.

    Returns:
        0 = DR
        1 = Glaucoma
        2 = AMD
        3 = Cataract
        4 = Normal
       -1 = Exclude / ambiguous
    """

    dr = to_binary(
        dr_value
    )

    glaucoma = to_binary(
        glaucoma_value
    )

    amd = to_binary(
        amd_value
    )

    cataract = to_binary(
        cataract_value
    )

    disease_risk = to_binary(
        disease_risk_value
    )

    supported_positive_classes = []

    if dr == 1:

        supported_positive_classes.append(
            RFMiD_DR
        )

    if glaucoma == 1:

        supported_positive_classes.append(
            RFMiD_GLAUCOMA
        )

    if amd == 1:

        supported_positive_classes.append(
            RFMiD_AMD
        )

    if cataract == 1:

        supported_positive_classes.append(
            RFMiD_CATARACT
        )

    # --------------------------------------------------------
    # No supported disease
    # --------------------------------------------------------

    if len(supported_positive_classes) == 0:

        # Disease_Risk=0 with no disease-specific label
        # is treated as Normal.
        if disease_risk == 0:

            return RFMiD_NORMAL

        # Disease_Risk=1 does NOT tell us which disease
        # is present, so do not invent a class.
        return -1

    # --------------------------------------------------------
    # Exactly one supported disease
    # --------------------------------------------------------

    if len(supported_positive_classes) == 1:

        return supported_positive_classes[0]

    # --------------------------------------------------------
    # Multiple supported diseases
    #
    # Current model is single-class CrossEntropy.
    # Therefore ambiguous multi-label records must be
    # excluded instead of assigning an arbitrary disease.
    # --------------------------------------------------------

    return -1


# ============================================================
# DATASET
# ============================================================

class RFMiDDataset(
    Dataset
):

    def __init__(
        self,
        samples
    ):

        self.samples = samples

        self.mean = torch.tensor(
            MEAN,
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

        self.std = torch.tensor(
            STD,
            dtype=torch.float32
        ).view(
            3,
            1,
            1
        )

    def __len__(
        self
    ):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index
    ):

        image_path, label = (
            self.samples[index]
        )

        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if image is None:

            raise RuntimeError(
                "\nCould not read RFMiD image:\n"
                f"{image_path}"
            )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Convert to float
        # ----------------------------------------------------

        image = (
            image.astype(
                np.float32
            )
            / 255.0
        )

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        image = np.ascontiguousarray(
            image,
            dtype=np.float32
        )

        image = torch.from_numpy(
            image
        )

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        image = (
            image - self.mean
        ) / self.std

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return (
            image,
            label
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n")
    print("=" * 70)
    print("LOADING RFMID DATA")
    print("=" * 70)

    # --------------------------------------------------------
    # Read CSV
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH
    )

    print(
        "CSV rows:",
        len(df)
    )

    print("\nCSV columns:")

    for column in df.columns:

        print(
            " -",
            column
        )

    # --------------------------------------------------------
    # Image column
    # --------------------------------------------------------

    image_column = find_image_column(
        df.columns
    )

    if image_column is None:

        raise RuntimeError(
            "\nCould not identify RFMiD image column.\n"
            f"Available columns: {list(df.columns)}"
        )

    # --------------------------------------------------------
    # Disease columns
    # --------------------------------------------------------

    disease_columns = find_disease_columns(
        df.columns
    )

    print("\n")
    print("=" * 70)
    print("RFMID DISEASE COLUMN DETECTION")
    print("=" * 70)

    print(
        "Image column:",
        image_column
    )

    print(
        "DR column:",
        disease_columns["dr"]
    )

    print(
        "Glaucoma column:",
        disease_columns["glaucoma"]
    )

    print(
        "AMD column:",
        disease_columns["amd"]
    )

    print(
        "Cataract column:",
        disease_columns["cataract"]
    )

    print(
        "Disease_Risk column:",
        disease_columns["disease_risk"]
    )

    # --------------------------------------------------------
    # We require explicit disease columns.
    # --------------------------------------------------------

    explicit_columns = [
        disease_columns["dr"],
        disease_columns["glaucoma"],
        disease_columns["amd"],
        disease_columns["cataract"]
    ]

    explicit_count = sum(
        column is not None
        for column in explicit_columns
    )

    if explicit_count == 0:

        raise RuntimeError(
            "\nNo supported RFMiD disease-specific "
            "columns were detected.\n\n"
            "Disease_Risk alone cannot be converted "
            "into Diabetic Retinopathy."
        )

    # --------------------------------------------------------
    # Image lookup
    # --------------------------------------------------------

    image_lookup = build_image_lookup()

    if len(image_lookup) == 0:

        raise RuntimeError(
            "\nNo RFMiD images found in:\n"
            f"{DATA_DIR}"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    samples = []

    missing_images = 0

    invalid_rows = 0

    excluded_ambiguous = 0

    excluded_unknown = 0

    class_counts = {
        0: 0,
        1: 0,
        2: 0,
        3: 0,
        4: 0
    }

    # --------------------------------------------------------
    # Process rows
    # --------------------------------------------------------

    for _, row in df.iterrows():

        try:

            image_key = normalize_image_name(
                row[image_column]
            )

            image_path = image_lookup.get(
                image_key
            )

            if image_path is None:

                missing_images += 1

                continue

            # ------------------------------------------------
            # Values
            # ------------------------------------------------

            dr_value = (
                row[disease_columns["dr"]]
                if disease_columns["dr"] is not None
                else 0
            )

            glaucoma_value = (
                row[disease_columns["glaucoma"]]
                if disease_columns["glaucoma"] is not None
                else 0
            )

            amd_value = (
                row[disease_columns["amd"]]
                if disease_columns["amd"] is not None
                else 0
            )

            cataract_value = (
                row[disease_columns["cataract"]]
                if disease_columns["cataract"] is not None
                else 0
            )

            disease_risk_value = (
                row[disease_columns["disease_risk"]]
                if disease_columns["disease_risk"] is not None
                else 0
            )

            # ------------------------------------------------
            # Unified label
            # ------------------------------------------------

            label = create_rfid_label(
                dr_value=dr_value,
                glaucoma_value=glaucoma_value,
                amd_value=amd_value,
                cataract_value=cataract_value,
                disease_risk_value=disease_risk_value
            )

            # ------------------------------------------------
            # Excluded
            # ------------------------------------------------

            if label == -1:

                supported_count = sum(
                    [
                        to_binary(dr_value),
                        to_binary(glaucoma_value),
                        to_binary(amd_value),
                        to_binary(cataract_value)
                    ]
                )

                if supported_count > 1:

                    excluded_ambiguous += 1

                else:

                    excluded_unknown += 1

                continue

            # ------------------------------------------------
            # Save sample
            # ------------------------------------------------

            samples.append(
                (
                    image_path,
                    label
                )
            )

            class_counts[label] += 1

        except Exception as error:

            invalid_rows += 1

            print(
                "\nWARNING: invalid RFMiD row:"
            )

            print(
                repr(error)
            )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RFMID DATA PREPARATION")
    print("=" * 70)

    print(
        "Matched usable samples:",
        len(samples)
    )

    print(
        "Missing images:",
        missing_images
    )

    print(
        "Invalid rows:",
        invalid_rows
    )

    print(
        "Excluded ambiguous multi-disease rows:",
        excluded_ambiguous
    )

    print(
        "Excluded unsupported/unknown rows:",
        excluded_unknown
    )

    print("\nUnified class distribution:")

    for class_id in range(NUM_CLASSES):

        print(
            f"{class_id}: "
            f"{LABELS[class_id]:25s} -> "
            f"{class_counts[class_id]}"
        )

    if len(samples) == 0:

        raise RuntimeError(
            "\nNo valid RFMiD samples were created."
        )

    # --------------------------------------------------------
    # Balance / limit each class
    # --------------------------------------------------------

    grouped = {
        class_id: []
        for class_id in range(NUM_CLASSES)
    }

    for sample in samples:

        image_path, label = sample

        grouped[label].append(
            sample
        )

    balanced_samples = []

    print("\n")
    print("=" * 70)
    print("RFMID CLASS LIMITING")
    print("=" * 70)

    for class_id in range(NUM_CLASSES):

        class_samples = grouped[class_id]

        random.shuffle(
            class_samples
        )

        available = len(
            class_samples
        )

        if (
            MAX_SAMPLES_PER_CLASS is not None
            and
            available > MAX_SAMPLES_PER_CLASS
        ):

            selected = class_samples[
                :MAX_SAMPLES_PER_CLASS
            ]

        else:

            selected = class_samples

        balanced_samples.extend(
            selected
        )

        print(
            f"{LABELS[class_id]:25s} | "
            f"Available: {available:4d} | "
            f"Selected: {len(selected):4d}"
        )

    samples = balanced_samples

    random.shuffle(
        samples
    )

    # --------------------------------------------------------
    # Final distribution
    # --------------------------------------------------------

    final_distribution = {
        class_id: 0
        for class_id in range(NUM_CLASSES)
    }

    for _, label in samples:

        final_distribution[label] += 1

    print("\n")
    print("=" * 70)
    print("FINAL RFMID UNIFIED DISTRIBUTION")
    print("=" * 70)

    for class_id in range(NUM_CLASSES):

        print(
            f"{class_id}: "
            f"{LABELS[class_id]:25s} -> "
            f"{final_distribution[class_id]} images"
        )

    print(
        "\nTotal RFMiD samples used:",
        len(samples)
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    split_index = int(
        len(samples)
        * TRAIN_RATIO
    )

    if split_index <= 0:

        raise RuntimeError(
            "RFMiD training split is empty."
        )

    if split_index >= len(samples):

        raise RuntimeError(
            "RFMiD validation split is empty."
        )

    train_samples = samples[
        :split_index
    ]

    val_samples = samples[
        split_index:
    ]

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = RFMiDDataset(
        train_samples
    )

    val_dataset = RFMiDDataset(
        val_samples
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Final information
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RFMID TRAIN / VALIDATION SPLIT")
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
        len(train_dataset),
        len(val_dataset)
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

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            images
        )

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        batch_size = images.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += batch_size

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

            batch_size = images.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += batch_size

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

class RFMiDClient(
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
        print("INITIALIZING RFMID FLOWER CLIENT")
        print("=" * 70)

        self.model = Net().to(
            DEVICE
        )

        self.train_loader = (
            train_loader
        )

        self.val_loader = (
            val_loader
        )

        self.train_size = (
            train_size
        )

        self.val_size = (
            val_size
        )

        self.criterion = (
            nn.CrossEntropyLoss()
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        print(
            "Device:",
            DEVICE
        )

        print(
            "Model: Current 5-class Net"
        )

        print(
            "Output classes:",
            NUM_CLASSES
        )

        print("\nGlobal label mapping:")

        for index, label in enumerate(
            LABELS
        ):

            print(
                f"  {index} -> {label}"
            )

        # ----------------------------------------------------
        # Federated parameter check
        # ----------------------------------------------------

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

        if len(parameters) != 10:

            raise RuntimeError(
                "\nExpected 10 federated tensors "
                f"but current model has {len(parameters)}.\n"
                "Check model.py."
            )

        for index, parameter in enumerate(
            parameters
        ):

            print(
                f"Tensor {index}: "
                f"{parameter.shape}"
            )

        print("=" * 70)

        print(
            "RFMID CLIENT READY"
        )

        print("=" * 70)

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
        print("=" * 70)
        print(">>> RFMID LOCAL TRAINING")
        print("=" * 70)

        # ----------------------------------------------------
        # Verify parameter count
        # ----------------------------------------------------

        expected = len(
            self.model.get_shared_parameters()
        )

        if len(parameters) != expected:

            raise RuntimeError(
                "\nFederated parameter count mismatch!\n"
                f"Server sent: {len(parameters)}\n"
                f"Model expects: {expected}"
            )

        # ----------------------------------------------------
        # Load global parameters
        # ----------------------------------------------------

        self.model.set_shared_parameters(
            parameters
        )

        # ----------------------------------------------------
        # Local epochs
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

        print(
            "Training samples:",
            self.train_size
        )

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        optimizer = optim.Adam(
            self.model.parameters(),
            lr=LEARNING_RATE
        )

        last_loss = 0.0

        last_accuracy = 0.0

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        for epoch in range(
            local_epochs
        ):

            loss, accuracy = train_model(
                self.model,
                self.train_loader,
                optimizer,
                self.criterion
            )

            last_loss = loss

            last_accuracy = accuracy

            print(
                f"Epoch "
                f"{epoch + 1}/{local_epochs} | "
                f"Loss: {loss:.4f} | "
                f"Accuracy: {accuracy:.4f}"
            )

        # ----------------------------------------------------
        # Return parameters
        # ----------------------------------------------------

        updated_parameters = (
            self.model.get_shared_parameters()
        )

        print(
            "\nReturning federated tensors:",
            len(updated_parameters)
        )

        print(
            "Final local loss:",
            f"{last_loss:.4f}"
        )

        print(
            "Final local accuracy:",
            f"{last_accuracy:.4f}"
        )

        print("=" * 70)
        print(">>> RFMID LOCAL TRAINING FINISHED")
        print("=" * 70)

        del optimizer

        gc.collect()

        return (
            updated_parameters,
            self.train_size,
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

        print("\n")
        print("=" * 70)
        print(">>> RFMID EVALUATION")
        print("=" * 70)

        expected = len(
            self.model.get_shared_parameters()
        )

        if len(parameters) != expected:

            raise RuntimeError(
                "\nFederated parameter count mismatch!\n"
                f"Server sent: {len(parameters)}\n"
                f"Model expects: {expected}"
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
            "RFMiD Validation Loss:",
            f"{loss:.4f}"
        )

        print(
            "RFMiD Validation Accuracy:",
            f"{accuracy:.4f}"
        )

        print(
            "Validation samples:",
            self.val_size
        )

        print("=" * 70)

        gc.collect()

        return (
            float(loss),
            self.val_size,
            {
                "val_loss": float(loss),
                "val_accuracy": float(accuracy)
            }
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("RFMID FLOWER CLIENT")
    print("=" * 70)

    set_seed()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        train_size,
        val_size
    ) = load_data()

    # --------------------------------------------------------
    # Create client
    # --------------------------------------------------------

    client = RFMiDClient(
        train_loader=train_loader,
        val_loader=val_loader,
        train_size=train_size,
        val_size=val_size
    )

    # --------------------------------------------------------
    # Final parameter check
    # --------------------------------------------------------

    parameter_count = len(
        client.model.get_shared_parameters()
    )

    print("\n")
    print("=" * 70)
    print("RFMID CLIENT SUMMARY")
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
        "Classes:",
        NUM_CLASSES
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
        "Local epochs:",
        LOCAL_EPOCHS
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Connect to Flower
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("CONNECTING RFMID CLIENT TO FLOWER SERVER")
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