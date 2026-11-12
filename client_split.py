import os
import random
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import flwr as fl

from model import Net
from label_mapping import (
    NUM_CLASSES,
    LABELS,
    map_split_label
)


# ============================================================
# CPU / REPRODUCIBILITY SETTINGS
# ============================================================

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

cv2.setNumThreads(0)

torch.set_num_threads(1)

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_NAME = "split"

DATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "classification",
    DATASET_NAME
)

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

TEST_DIR = os.path.join(
    DATA_DIR,
    "test"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

IMG_SIZE = 224

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

EPOCHS = 2

VALIDATION_SPLIT = 0.20

FEDPROX_MU = 0.01

SERVER_ADDRESS = "127.0.0.1:9090"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# GLOBAL UNIFIED CLASSES
# ============================================================

EXPECTED_NUM_CLASSES = 5

if NUM_CLASSES != EXPECTED_NUM_CLASSES:

    raise ValueError(
        "\nLABEL MAPPING ERROR!\n"
        f"Expected NUM_CLASSES = 5\n"
        f"Found NUM_CLASSES = {NUM_CLASSES}"
    )


print("\n" + "=" * 70)
print("STARTING SPLIT FEDERATED CLIENT")
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
    "Train directory:",
    TRAIN_DIR
)

print(
    "Test directory:",
    TEST_DIR
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
    EPOCHS
)

print(
    "FedProx mu:",
    FEDPROX_MU
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
# IMAGE EXTENSIONS
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
)


# ============================================================
# NORMALIZATION
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.ToPILImage(),

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
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


eval_transform = transforms.Compose([

    transforms.ToPILImage(),

    transforms.Resize(
        (IMG_SIZE, IMG_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# FOLDER NAME NORMALIZATION
# ============================================================

def normalize_folder_name(name):

    name = str(name).strip().lower()

    name = name.replace(
        "_",
        " "
    )

    name = name.replace(
        "-",
        " "
    )

    name = " ".join(
        name.split()
    )

    return name


# ============================================================
# MAP SPLIT FOLDER TO GLOBAL CLASS
# ============================================================

def get_split_class_index(folder_name):

    normalized = normalize_folder_name(
        folder_name
    )

    index = map_split_label(
        normalized
    )

    return index


# ============================================================
# DISCOVER DATASET CLASSES
# ============================================================

def discover_class_folders(directory):

    if not os.path.isdir(directory):

        raise FileNotFoundError(
            f"\nDIRECTORY NOT FOUND:\n{directory}"
        )

    folders = []

    for item in os.listdir(directory):

        full_path = os.path.join(
            directory,
            item
        )

        if os.path.isdir(full_path):

            folders.append(item)

    folders.sort()

    return folders


# ============================================================
# PRINT DATASET STRUCTURE
# ============================================================

def validate_dataset_structure():

    print("\n" + "=" * 70)
    print("VALIDATING SPLIT DATASET")
    print("=" * 70)

    print(
        "\nDataset directory:"
    )

    print(
        DATA_DIR
    )

    print(
        "\nTrain directory:"
    )

    print(
        TRAIN_DIR
    )

    print(
        "\nTest directory:"
    )

    print(
        TEST_DIR
    )

    # --------------------------------------------------------
    # ROOT
    # --------------------------------------------------------

    if not os.path.isdir(DATA_DIR):

        raise FileNotFoundError(
            "\nSPLIT DATASET DIRECTORY NOT FOUND:\n"
            f"{DATA_DIR}"
        )

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    if not os.path.isdir(TRAIN_DIR):

        raise FileNotFoundError(
            "\nTRAIN DIRECTORY NOT FOUND:\n"
            f"{TRAIN_DIR}"
        )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    if not os.path.isdir(TEST_DIR):

        raise FileNotFoundError(
            "\nTEST DIRECTORY NOT FOUND:\n"
            f"{TEST_DIR}"
        )

    # --------------------------------------------------------
    # TRAIN FOLDERS
    # --------------------------------------------------------

    train_folders = discover_class_folders(
        TRAIN_DIR
    )

    # --------------------------------------------------------
    # TEST FOLDERS
    # --------------------------------------------------------

    test_folders = discover_class_folders(
        TEST_DIR
    )

    print(
        "\nDetected TRAIN folders:"
    )

    for folder in train_folders:

        print(
            f"  {folder}"
        )

    print(
        "\nDetected TEST folders:"
    )

    for folder in test_folders:

        print(
            f"  {folder}"
        )

    # --------------------------------------------------------
    # REQUIRED GLOBAL CLASSES
    # --------------------------------------------------------

    required_classes = {
        "cataract",
        "diabetic retinopathy",
        "glaucoma",
        "normal"
    }

    train_normalized = {
        normalize_folder_name(folder)
        for folder in train_folders
    }

    test_normalized = {
        normalize_folder_name(folder)
        for folder in test_folders
    }

    missing_train = (
        required_classes
        - train_normalized
    )

    missing_test = (
        required_classes
        - test_normalized
    )

    if missing_train:

        raise ValueError(
            "\nTRAIN DATASET IS MISSING CLASSES:\n"
            f"{sorted(missing_train)}"
        )

    if missing_test:

        raise ValueError(
            "\nTEST DATASET IS MISSING CLASSES:\n"
            f"{sorted(missing_test)}"
        )

    print(
        "\nDataset structure validation PASSED."
    )


# ============================================================
# DISCOVER IMAGE FILES
# ============================================================

def discover_samples(directory):

    samples = []

    class_counts = {
        index: 0
        for index in range(NUM_CLASSES)
    }

    excluded_folders = []

    folders = discover_class_folders(
        directory
    )

    for folder in folders:

        class_index = get_split_class_index(
            folder
        )

        if class_index < 0:

            excluded_folders.append(
                folder
            )

            continue

        folder_path = os.path.join(
            directory,
            folder
        )

        for root, _, files in os.walk(
            folder_path
        ):

            for file in files:

                if not file.lower().endswith(
                    IMAGE_EXTENSIONS
                ):
                    continue

                image_path = os.path.join(
                    root,
                    file
                )

                samples.append(
                    (
                        image_path,
                        class_index
                    )
                )

                class_counts[
                    class_index
                ] += 1

    if excluded_folders:

        print(
            "\nExcluded folders:"
        )

        for folder in excluded_folders:

            print(
                f"  {folder}"
            )

    return (
        samples,
        class_counts
    )


# ============================================================
# CUSTOM DATASET
# ============================================================

class SplitDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples

        self.transform = transform

    def __len__(self):

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
            image_path
        )

        if image is None:

            raise RuntimeError(
                "\nFAILED TO READ IMAGE:\n"
                f"{image_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform is not None:

            image = self.transform(
                image
            )

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# ============================================================
# STRATIFIED TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    samples
):

    class_samples = {
        index: []
        for index in range(NUM_CLASSES)
    }

    for sample in samples:

        image_path, label = sample

        class_samples[
            label
        ].append(
            (
                image_path,
                label
            )
        )

    train_samples = []

    val_samples = []

    for class_index in range(
        NUM_CLASSES
    ):

        current_samples = (
            class_samples[
                class_index
            ]
        )

        random.shuffle(
            current_samples
        )

        total = len(
            current_samples
        )

        if total == 0:

            continue

        val_count = int(
            total *
            VALIDATION_SPLIT
        )

        # Make sure validation has at least
        # one image when possible.
        if total >= 2 and val_count < 1:

            val_count = 1

        if (
            total >= 2
            and val_count >= total
        ):

            val_count = total - 1

        class_val = (
            current_samples[
                :val_count
            ]
        )

        class_train = (
            current_samples[
                val_count:
            ]
        )

        val_samples.extend(
            class_val
        )

        train_samples.extend(
            class_train
        )

    random.shuffle(
        train_samples
    )

    random.shuffle(
        val_samples
    )

    return (
        train_samples,
        val_samples
    )


# ============================================================
# PRINT CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    samples,
    title
):

    counts = np.zeros(
        NUM_CLASSES,
        dtype=int
    )

    for _, label in samples:

        if 0 <= label < NUM_CLASSES:

            counts[label] += 1

    print(
        f"\n{title}"
    )

    for index in range(
        NUM_CLASSES
    ):

        print(
            f"  {index} - "
            f"{LABELS[index]:<25} : "
            f"{counts[index]}"
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\n" + "=" * 70)
    print("LOADING SPLIT DATASET")
    print("=" * 70)

    validate_dataset_structure()

    # --------------------------------------------------------
    # TRAIN SOURCE
    # --------------------------------------------------------

    (
        train_source_samples,
        train_source_counts
    ) = discover_samples(
        TRAIN_DIR
    )

    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    (
        test_samples,
        test_counts
    ) = discover_samples(
        TEST_DIR
    )

    print(
        "\nTRAIN SOURCE DISTRIBUTION:"
    )

    for index in range(
        NUM_CLASSES
    ):

        print(
            f"  {index} - "
            f"{LABELS[index]:<25} : "
            f"{train_source_counts[index]}"
        )

    print(
        "\nTEST DISTRIBUTION:"
    )

    for index in range(
        NUM_CLASSES
    ):

        print(
            f"  {index} - "
            f"{LABELS[index]:<25} : "
            f"{test_counts[index]}"
        )

    # --------------------------------------------------------
    # CHECK TRAIN SOURCE
    # --------------------------------------------------------

    if len(
        train_source_samples
    ) == 0:

        raise ValueError(
            "\nNO TRAINING IMAGES FOUND."
        )

    # --------------------------------------------------------
    # CHECK TEST
    # --------------------------------------------------------

    if len(
        test_samples
    ) == 0:

        raise ValueError(
            "\nNO TEST IMAGES FOUND."
        )

    # --------------------------------------------------------
    # CREATE VALIDATION FROM TRAIN
    # --------------------------------------------------------

    (
        train_samples,
        val_samples
    ) = create_train_validation_split(
        train_source_samples
    )

    # --------------------------------------------------------
    # PRINT FINAL DISTRIBUTIONS
    # --------------------------------------------------------

    print_distribution(
        train_samples,
        "\nFINAL TRAIN DISTRIBUTION:"
    )

    print_distribution(
        val_samples,
        "\nFINAL VALIDATION DISTRIBUTION:"
    )

    print_distribution(
        test_samples,
        "\nFINAL TEST DISTRIBUTION:"
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = SplitDataset(
        train_samples,
        transform=train_transform
    )

    val_dataset = SplitDataset(
        val_samples,
        transform=eval_transform
    )

    test_dataset = SplitDataset(
        test_samples,
        transform=eval_transform
    )

    # --------------------------------------------------------
    # DATALOADERS
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT DATASET SUMMARY")
    print("=" * 70)

    print(
        "\nTraining images    :",
        len(train_dataset)
    )

    print(
        "Validation images  :",
        len(val_dataset)
    )

    print(
        "Testing images     :",
        len(test_dataset)
    )

    print(
        "Total images       :",
        len(train_dataset)
        + len(val_dataset)
        + len(test_dataset)
    )

    print(
        "Number of classes  :",
        NUM_CLASSES
    )

    print(
        "\nDataset loading PASSED."
    )

    return (
        train_loader,
        val_loader,
        test_loader
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    print("\n" + "=" * 70)
    print("INITIALIZING CURRENT FEDERATED MODEL")
    print("=" * 70)

    print(
        "Using AlexNet backbone"
    )

    print(
        "Model: Current 5-class Net"
    )

    print(
        "Device:",
        DEVICE
    )

    print(
        "Output classes:",
        NUM_CLASSES
    )

    model = Net()

    model = model.to(
        DEVICE
    )

    # --------------------------------------------------------
    # FEDERATED PARAMETER CHECK
    # --------------------------------------------------------

    parameters = (
        model.get_shared_parameters()
    )

    print(
        "\nFederated tensors:",
        len(parameters)
    )

    for index, parameter in enumerate(
        parameters
    ):

        print(
            f"Tensor {index}: "
            f"{parameter.shape}"
        )

    if len(parameters) != 10:

        raise ValueError(
            "\nFEDERATED PARAMETER COUNT ERROR!\n"
            f"Expected: 10\n"
            f"Found: {len(parameters)}"
        )

    print(
        "\nCurrent model parameter check PASSED."
    )

    return model


# ============================================================
# TRAIN ONE LOCAL ROUND
# ============================================================

def train_model(
    model,
    train_loader,
    global_parameters
):

    model.train()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ],
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # GLOBAL PARAMETERS FOR FEDPROX
    # --------------------------------------------------------

    global_tensors = []

    for parameter in global_parameters:

        tensor = torch.tensor(
            parameter,
            dtype=torch.float32,
            device=DEVICE
        )

        global_tensors.append(
            tensor
        )

    final_loss = 0.0

    for epoch in range(
        EPOCHS
    ):

        running_loss = 0.0

        running_ce_loss = 0.0

        running_prox_loss = 0.0

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

            ce_loss = criterion(
                outputs,
                labels
            )

            # ------------------------------------------------
            # FEDPROX
            # ------------------------------------------------

            current_parameters = [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad
            ]

            prox_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

            if len(
                current_parameters
            ) != len(
                global_tensors
            ):

                raise ValueError(
                    "\nFedProx parameter mismatch!\n"
                    f"Current: "
                    f"{len(current_parameters)}\n"
                    f"Global: "
                    f"{len(global_tensors)}"
                )

            for current_parameter, global_parameter in zip(
                current_parameters,
                global_tensors
            ):

                prox_loss += torch.sum(
                    (
                        current_parameter
                        - global_parameter
                    ) ** 2
                )

            prox_loss = (
                FEDPROX_MU
                / 2.0
            ) * prox_loss

            loss = (
                ce_loss
                + prox_loss
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # METRICS
            # ------------------------------------------------

            running_loss += (
                loss.item()
            )

            running_ce_loss += (
                ce_loss.item()
            )

            running_prox_loss += (
                prox_loss.item()
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += (
                labels.size(0)
            )

            correct += (
                predicted == labels
            ).sum().item()

        final_loss = (
            running_loss
            / len(train_loader)
        )

        average_ce = (
            running_ce_loss
            / len(train_loader)
        )

        average_prox = (
            running_prox_loss
            / len(train_loader)
        )

        accuracy = (
            100.0
            * correct
            / total
            if total > 0
            else 0.0
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {final_loss:.4f} "
            f"CE: {average_ce:.4f} "
            f"FedProx: {average_prox:.4f} "
            f"Accuracy: {accuracy:.2f}%"
        )

    return final_loss


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    model,
    data_loader,
    dataset_name
):

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0

    correct = 0

    total = 0

    class_correct = np.zeros(
        NUM_CLASSES,
        dtype=int
    )

    class_total = np.zeros(
        NUM_CLASSES,
        dtype=int
    )

    with torch.no_grad():

        for images, labels in data_loader:

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
                * labels.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            total += (
                labels.size(0)
            )

            correct += (
                predicted == labels
            ).sum().item()

            for label, prediction in zip(
                labels,
                predicted
            ):

                label_index = (
                    int(label.item())
                )

                if 0 <= label_index < NUM_CLASSES:

                    class_total[
                        label_index
                    ] += 1

                    if (
                        int(prediction.item())
                        == label_index
                    ):

                        class_correct[
                            label_index
                        ] += 1

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

    print(
        f"\n{dataset_name} Loss: "
        f"{average_loss:.4f}"
    )

    print(
        f"{dataset_name} Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"\n{dataset_name} class accuracy:"
    )

    for index in range(
        NUM_CLASSES
    ):

        if class_total[index] > 0:

            class_accuracy = (
                100.0
                * class_correct[index]
                / class_total[index]
            )

            print(
                f"  {index} - "
                f"{LABELS[index]:<25} : "
                f"{class_accuracy:.2f}% "
                f"({class_correct[index]}/"
                f"{class_total[index]})"
            )

        else:

            print(
                f"  {index} - "
                f"{LABELS[index]:<25} : "
                f"N/A"
            )

    return (
        average_loss,
        accuracy,
        total
    )


# ============================================================
# FLOWER CLIENT
# ============================================================

class SplitClient(
    fl.client.NumPyClient
):

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader
    ):

        self.model = model

        self.train_loader = train_loader

        self.val_loader = val_loader

        self.test_loader = test_loader

    # --------------------------------------------------------
    # GET PARAMETERS
    # --------------------------------------------------------

    def get_parameters(
        self,
        config
    ):

        return (
            self.model
            .get_shared_parameters()
        )

    # --------------------------------------------------------
    # SET PARAMETERS
    # --------------------------------------------------------

    def set_parameters(
        self,
        parameters
    ):

        if len(parameters) != 10:

            raise ValueError(
                "\nFEDERATED PARAMETER COUNT MISMATCH!\n"
                f"Received: {len(parameters)}\n"
                f"Expected: 10"
            )

        self.model.set_shared_parameters(
            parameters
        )

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(
        self,
        parameters,
        config
    ):

        print("\n" + "=" * 70)
        print("FEDERATED TRAINING - SPLIT CLIENT")
        print("=" * 70)

        # ----------------------------------------------------
        # LOAD GLOBAL PARAMETERS
        # ----------------------------------------------------

        self.set_parameters(
            parameters
        )

        # ----------------------------------------------------
        # KEEP COPY FOR FEDPROX
        # ----------------------------------------------------

        global_parameters = [
            np.copy(parameter)
            for parameter in parameters
        ]

        # ----------------------------------------------------
        # LOCAL TRAINING
        # ----------------------------------------------------

        loss = train_model(
            self.model,
            self.train_loader,
            global_parameters
        )

        # ----------------------------------------------------
        # LOCAL VALIDATION
        # ----------------------------------------------------

        print(
            "\nLocal validation:"
        )

        (
            val_loss,
            val_accuracy,
            _
        ) = evaluate_model(
            self.model,
            self.val_loader,
            "SPLIT Validation"
        )

        num_examples = len(
            self.train_loader.dataset
        )

        print(
            "\nSPLIT local training completed."
        )

        print(
            f"Training samples: "
            f"{num_examples}"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        # ----------------------------------------------------
        # RETURN FEDERATED PARAMETERS
        # ----------------------------------------------------

        updated_parameters = (
            self.get_parameters(
                config
            )
        )

        if len(
            updated_parameters
        ) != 10:

            raise ValueError(
                "\nUPDATED PARAMETER COUNT ERROR!\n"
                f"Expected: 10\n"
                f"Found: "
                f"{len(updated_parameters)}"
            )

        return (
            updated_parameters,
            num_examples,
            {
                "loss": float(loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(
                    val_accuracy
                )
            }
        )

    # --------------------------------------------------------
    # EVALUATE
    # --------------------------------------------------------

    def evaluate(
        self,
        parameters,
        config
    ):

        print("\n" + "=" * 70)
        print("FEDERATED EVALUATION - SPLIT CLIENT")
        print("=" * 70)

        self.set_parameters(
            parameters
        )

        (
            loss,
            accuracy,
            num_examples
        ) = evaluate_model(
            self.model,
            self.test_loader,
            "SPLIT Test"
        )

        return (
            float(loss),
            num_examples,
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

    print("\n" + "=" * 70)
    print("INITIALIZING SPLIT FLOWER CLIENT")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Dataset:",
        DATASET_NAME
    )

    print(
        "Using current AlexNet-based Net"
    )

    print(
        "Number of classes:",
        NUM_CLASSES
    )

    print("\nGlobal label mapping:")

    for index, label in enumerate(
        LABELS
    ):

        print(
            f"  {index} -> {label}"
        )

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader
    ) = load_data()

    # --------------------------------------------------------
    # CREATE CURRENT MODEL
    # --------------------------------------------------------

    model = create_model()

    # --------------------------------------------------------
    # CREATE FLOWER CLIENT
    # --------------------------------------------------------

    client = SplitClient(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader
    )

    # --------------------------------------------------------
    # FINAL CLIENT SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT CLIENT READY")
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
        len(
            model.get_shared_parameters()
        )
    )

    print(
        "Training samples:",
        len(
            train_loader.dataset
        )
    )

    print(
        "Validation samples:",
        len(
            val_loader.dataset
        )
    )

    print(
        "Test samples:",
        len(
            test_loader.dataset
        )
    )

    print(
        "Local epochs:",
        EPOCHS
    )

    # --------------------------------------------------------
    # CONNECT TO FLOWER
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CONNECTING SPLIT CLIENT TO FLOWER SERVER")
    print("=" * 70)

    print(
        "Server:",
        SERVER_ADDRESS
    )

    print(
        "\nMake sure server.py is already running."
    )

    print(
        "Waiting for federated rounds..."
    )

    # --------------------------------------------------------
    # MODERN FLOWER API
    # --------------------------------------------------------

    fl.client.start_client(
        server_address=SERVER_ADDRESS,
        client=client.to_client()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()