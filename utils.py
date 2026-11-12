import torch
import cv2
import numpy as np


# ============================================================
# IMAGE SIZE
# ============================================================

IMG_SIZE = 224


# ============================================================
# CLASS LABELS
# ============================================================

CLASS_LABELS = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "AMD",
    "Cataract",
    "Normal"
]


# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(outputs, labels):

    _, predicted = torch.max(
        outputs,
        1
    )

    correct = (
        predicted == labels
    ).sum().item()

    return correct / len(labels)


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(
    model,
    path="results/model.pth"
):

    torch.save(
        model.state_dict(),
        path
    )

    print(
        f"Model saved at {path}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(
    model,
    path="results/model.pth",
    device="cpu"
):

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    model.eval()

    print(
        f"Model loaded from {path}"
    )

    return model


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(img_path):

    img = cv2.imread(
        img_path
    )

    if img is None:

        print(
            "Error: Image not found!"
        )

        return None


    # Resize
    img = cv2.resize(
        img,
        (IMG_SIZE, IMG_SIZE)
    )


    # Gaussian blur
    img = cv2.GaussianBlur(
        img,
        (5, 5),
        0
    )


    # Normalize
    img = img / 255.0


    # HWC -> CHW
    img = np.transpose(
        img,
        (2, 0, 1)
    )


    return torch.tensor(
        img,
        dtype=torch.float32
    )


# ============================================================
# PREDICT
# ============================================================

def predict(
    model,
    img_path,
    device="cpu"
):

    img = preprocess_image(
        img_path
    )


    if img is None:

        return (
            "Invalid Image",
            0.0
        )


    img = (
        img
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        outputs = model(
            img
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )


    predicted_index = (
        predicted.item()
    )


    confidence = round(
        confidence.item() * 100,
        2
    )


    predicted_label = CLASS_LABELS[
        predicted_index
    ]


    return (
        predicted_label,
        confidence
    )