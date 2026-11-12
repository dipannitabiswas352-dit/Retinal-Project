# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_NAME = "alexnet"

IMG_SIZE = 224

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

EPOCHS = 2


# ============================================================
# RETINAL CONDITION CLASSES
# ============================================================

NUM_CLASSES = 5

LABELS = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "AMD",
    "Cataract",
    "Normal"
]