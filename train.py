import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from model import Net


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
EYEPACS_DIR = os.path.join(DATA_DIR, "classification", "EyePACS")

TRAIN_DIR = os.path.join(EYEPACS_DIR, "train")
VALID_DIR = os.path.join(EYEPACS_DIR, "valid")
TEST_DIR = os.path.join(EYEPACS_DIR, "test")

TRAIN_CSV = os.path.join(EYEPACS_DIR, "train.csv")
VALID_CSV = os.path.join(EYEPACS_DIR, "valid.csv")
TEST_CSV = os.path.join(EYEPACS_DIR, "test.csv")


# ============================================================
# CONFIG
# ============================================================

IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LR = 0.0005
NUM_CLASSES = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


# ============================================================
# START
# ============================================================

print("\n" + "=" * 70)
print("CENTRALIZED EYEPACS TRAINING")
print("=" * 70)

print("Device      :", DEVICE)
print("Dataset path:", EYEPACS_DIR)


# ============================================================
# CHECK DIRECTORIES
# ============================================================

for folder in [TRAIN_DIR, VALID_DIR, TEST_DIR]:
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"\nFolder not found:\n{folder}"
        )


# ============================================================
# FIND CSV
# ============================================================

def get_csv_path(split):

    local_path = os.path.join(
        EYEPACS_DIR,
        f"{split}.csv"
    )

    if os.path.exists(local_path):
        return local_path

    fallback_path = os.path.join(
        DATA_DIR,
        f"{split}.csv"
    )

    if os.path.exists(fallback_path):
        return fallback_path

    return None


# ============================================================
# LOAD CSV
# ============================================================

def load_csv(csv_path):

    print("\nReading CSV:")
    print(csv_path)

    df = pd.read_csv(csv_path)

    print("Rows   :", len(df))
    print("Columns:", list(df.columns))

    if "id_code" not in df.columns:
        raise ValueError(
            "CSV must contain 'id_code' column."
        )

    if "diagnosis" not in df.columns:
        raise ValueError(
            "CSV must contain 'diagnosis' column."
        )

    df["id_code"] = (
        df["id_code"]
        .astype(str)
        .str.strip()
    )

    df["diagnosis"] = pd.to_numeric(
        df["diagnosis"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["id_code", "diagnosis"]
    )

    df["diagnosis"] = (
        df["diagnosis"]
        .astype(int)
    )

    invalid = ~df["diagnosis"].isin(
        [0, 1, 2, 3, 4]
    )

    if invalid.any():
        bad = sorted(
            df.loc[
                invalid,
                "diagnosis"
            ].unique().tolist()
        )

        raise ValueError(
            f"Invalid diagnosis labels found: {bad}"
        )

    return df


# ============================================================
# GET IMAGE ID
# ============================================================

def get_image_id(filename):
    return os.path.splitext(filename)[0].strip()


# ============================================================
# LOAD ONE SPLIT
# ============================================================

def load_split(image_dir, csv_path, split_name):

    print("\n" + "=" * 70)
    print(f"LOADING {split_name.upper()}")
    print("=" * 70)

    if csv_path is None:
        raise FileNotFoundError(
            f"\nNo {split_name}.csv found.\n\n"
            f"Expected:\n"
            f"{os.path.join(EYEPACS_DIR, split_name + '.csv')}\n"
            f"or\n"
            f"{os.path.join(DATA_DIR, split_name + '.csv')}"
        )

    df = load_csv(csv_path)

    label_map = dict(
        zip(
            df["id_code"],
            df["diagnosis"]
        )
    )

    X = []
    y = []

    total_images = 0
    matched_images = 0
    missing_labels = 0
    unreadable_images = 0

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png"
    )

    for root, _, files in os.walk(image_dir):

        for filename in files:

            if not filename.lower().endswith(
                valid_extensions
            ):
                continue

            total_images += 1

            image_id = get_image_id(filename)

            if image_id not in label_map:
                missing_labels += 1
                continue

            image_path = os.path.join(
                root,
                filename
            )

            img = cv2.imread(image_path)

            if img is None:
                unreadable_images += 1
                continue

            img = cv2.resize(
                img,
                (IMG_SIZE, IMG_SIZE)
            )

            img = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2RGB
            )

            X.append(img)
            y.append(label_map[image_id])

            matched_images += 1

    print("\nImage statistics")
    print("-----------------------------")
    print("Total images found :", total_images)
    print("Images matched     :", matched_images)
    print("Missing labels     :", missing_labels)
    print("Unreadable images  :", unreadable_images)

    if matched_images == 0:
        raise ValueError(
            f"\nNO IMAGES MATCHED FOR {split_name.upper()}.\n\n"
            f"The CSV IDs do not match the image filenames in:\n"
            f"{image_dir}"
        )

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int64
    )

    print("\nLoaded shapes")
    print("X:", X.shape)
    print("y:", y.shape)

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    print("\nClass distribution")

    for cls, count in zip(unique, counts):
        print(
            f"Class {cls} ({CLASS_NAMES[cls]}) : {count}"
        )

    return X, y


# ============================================================
# CSV PATHS
# ============================================================

train_csv = get_csv_path("train")
valid_csv = get_csv_path("valid")
test_csv = get_csv_path("test")

print("\n" + "=" * 70)
print("CSV FILES")
print("=" * 70)

print("Train:", train_csv)
print("Valid:", valid_csv)
print("Test :", test_csv)


# ============================================================
# LOAD DATA
# ============================================================

X_train, y_train = load_split(
    TRAIN_DIR,
    train_csv,
    "train"
)

X_valid, y_valid = load_split(
    VALID_DIR,
    valid_csv,
    "valid"
)

X_test, y_test = load_split(
    TEST_DIR,
    test_csv,
    "test"
)


# ============================================================
# NORMALIZATION
# ============================================================

X_train /= 255.0
X_valid /= 255.0
X_test /= 255.0


# ============================================================
# HWC -> CHW
# ============================================================

X_train = np.transpose(
    X_train,
    (0, 3, 1, 2)
)

X_valid = np.transpose(
    X_valid,
    (0, 3, 1, 2)
)

X_test = np.transpose(
    X_test,
    (0, 3, 1, 2)
)


# ============================================================
# PYTORCH DATASETS
# ============================================================

train_dataset = TensorDataset(
    torch.tensor(
        X_train,
        dtype=torch.float32
    ),
    torch.tensor(
        y_train,
        dtype=torch.long
    )
)

valid_dataset = TensorDataset(
    torch.tensor(
        X_valid,
        dtype=torch.float32
    ),
    torch.tensor(
        y_valid,
        dtype=torch.long
    )
)

test_dataset = TensorDataset(
    torch.tensor(
        X_test,
        dtype=torch.float32
    ),
    torch.tensor(
        y_test,
        dtype=torch.long
    )
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

print("\nCreating CNN model...")

model = Net(
    num_classes=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LR
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    avg_loss = total_loss / max(
        len(loader),
        1
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    return avg_loss, accuracy


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING STARTED")
print("=" * 70)

results_dir = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(
    results_dir,
    exist_ok=True
)

best_valid_accuracy = 0.0

best_model_path = os.path.join(
    results_dir,
    "centralized_best_model.pth"
)


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        all_labels.extend(
            labels.detach()
            .cpu()
            .numpy()
        )

    train_loss = (
        total_loss /
        max(len(train_loader), 1)
    )

    train_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    valid_loss, valid_accuracy = evaluate(
        valid_loader
    )

    print("\n" + "-" * 60)
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print(f"Train Loss     : {train_loss:.4f}")
    print(f"Train Accuracy : {train_accuracy:.4f}")
    print(f"Valid Loss     : {valid_loss:.4f}")
    print(f"Valid Accuracy : {valid_accuracy:.4f}")

    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = valid_accuracy

        torch.save(
            model.state_dict(),
            best_model_path
        )

        print("Best model saved.")


# ============================================================
# LOAD BEST MODEL
# ============================================================

if os.path.exists(best_model_path):

    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=DEVICE
        )
    )

    print("\nBest validation model loaded.")


# ============================================================
# FINAL TEST
# ============================================================

test_loss, test_accuracy = evaluate(
    test_loader
)

print("\n" + "=" * 70)
print("FINAL TEST RESULT")
print("=" * 70)

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy:.4f}")


# ============================================================
# SAVE FINAL MODEL
# ============================================================

final_model_path = os.path.join(
    results_dir,
    "centralized_model.pth"
)

torch.save(
    model.state_dict(),
    final_model_path
)

print("\nFinal model saved:")
print(final_model_path)

print("\n" + "=" * 70)
print("CENTRALIZED TRAINING COMPLETED")
print("=" * 70)