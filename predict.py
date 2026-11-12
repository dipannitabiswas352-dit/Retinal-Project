# ============================================================
# predict.py
# RETINAL FEDERATED LEARNING - PREDICTION + GRAD-CAM
# ============================================================

import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
from PIL import Image

# ============================================================
# PROJECT PATH
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

# ============================================================
# IMPORT PROJECT CONFIG / MODEL
# ============================================================

from config import (
    IMG_SIZE,
    NUM_CLASSES,
    LABELS
)

from model import Net


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "results",
    "global_model.pth"
)

TEST_IMAGE = os.path.join(
    PROJECT_ROOT,
    "data",
    "segmentation",
    "STARE",
    "image",
    "40_training_3.png"
)


# ============================================================
# IMAGE NORMALIZATION
# ============================================================

MEAN = torch.tensor(
    [0.485, 0.456, 0.406],
    dtype=torch.float32
).view(1, 3, 1, 1)

STD = torch.tensor(
    [0.229, 0.224, 0.225],
    dtype=torch.float32
).view(1, 3, 1, 1)


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):
    """
    Load retinal image and convert it into
    a normalized tensor suitable for AlexNet.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    image = Image.open(image_path).convert("RGB")

    image = image.resize(
        (IMG_SIZE, IMG_SIZE),
        Image.Resampling.BILINEAR
    )

    image_array = np.asarray(
        image,
        dtype=np.float32
    ) / 255.0

    tensor = torch.from_numpy(
        image_array
    ).permute(
        2, 0, 1
    )

    tensor = tensor.unsqueeze(0)

    tensor = (
        tensor - MEAN
    ) / STD

    return tensor.to(DEVICE), image


# ============================================================
# LOAD FEDERATED MODEL
# ============================================================

def load_federated_model(model_path):
    """
    Load the current 5-class federated model.

    Supported formats:
    1. list/tuple of NumPy arrays
    2. dictionary containing parameters
    3. PyTorch state_dict
    4. dictionary containing state_dict
    """

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"\nGlobal model not found:\n{model_path}\n\n"
            "Make sure federated training has completed "
            "and results/global_model.pth exists."
        )

    print("\nLoading global model:")
    print(model_path)

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False
    )

    model = Net()

    # ========================================================
    # CASE 1: DIRECT LIST / TUPLE OF FEDERATED PARAMETERS
    # ========================================================

    if isinstance(checkpoint, (list, tuple)):

        print(
            f"Detected federated parameter list: "
            f"{len(checkpoint)} tensors"
        )

        model.set_shared_parameters(
            checkpoint
        )

    # ========================================================
    # CASE 2: DICTIONARY
    # ========================================================

    elif isinstance(checkpoint, dict):

        loaded = False

        # ----------------------------------------------------
        # POSSIBLE FEDERATED PARAMETER KEYS
        # ----------------------------------------------------

        possible_parameter_keys = [
            "parameters",
            "params",
            "weights",
            "shared_parameters"
        ]

        for key in possible_parameter_keys:

            if key in checkpoint:

                parameters = checkpoint[key]

                if isinstance(
                    parameters,
                    (list, tuple)
                ):

                    print(
                        f"Detected federated parameters "
                        f"under key '{key}': "
                        f"{len(parameters)} tensors"
                    )

                    model.set_shared_parameters(
                        parameters
                    )

                    loaded = True
                    break

        # ----------------------------------------------------
        # STATE DICT
        # ----------------------------------------------------

        if not loaded:

            possible_state_keys = [
                "state_dict",
                "model_state_dict",
                "model"
            ]

            for key in possible_state_keys:

                if key in checkpoint:

                    state = checkpoint[key]

                    if isinstance(state, dict):

                        print(
                            f"Detected state_dict "
                            f"under key '{key}'"
                        )

                        model.load_state_dict(
                            state,
                            strict=False
                        )

                        loaded = True
                        break

        # ----------------------------------------------------
        # DIRECT STATE DICT
        # ----------------------------------------------------

        if not loaded:

            if all(
                isinstance(k, str)
                for k in checkpoint.keys()
            ):

                try:

                    model.load_state_dict(
                        checkpoint,
                        strict=False
                    )

                    print(
                        "Detected direct PyTorch state_dict"
                    )

                    loaded = True

                except Exception:
                    loaded = False

        if not loaded:

            raise RuntimeError(
                "Unable to understand global_model.pth format.\n"
                f"Checkpoint type: {type(checkpoint)}\n"
                f"Checkpoint keys: "
                f"{list(checkpoint.keys())[:20]}"
            )

    else:

        raise RuntimeError(
            "Unsupported global model format:\n"
            f"{type(checkpoint)}"
        )

    model = model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# NORMAL PREDICTION
# ============================================================

def predict_image(model, input_tensor):
    """
    Perform normal classification prediction.
    """

    model.eval()

    with torch.no_grad():

        output = model(
            input_tensor
        )

        if output.ndim != 2:

            raise RuntimeError(
                "Unexpected model output shape: "
                f"{tuple(output.shape)}"
            )

        if output.shape[1] != NUM_CLASSES:

            raise RuntimeError(
                f"Model returned {output.shape[1]} classes, "
                f"but config expects {NUM_CLASSES}."
            )

        probabilities = F.softmax(
            output,
            dim=1
        )

        probabilities = probabilities[
            0
        ].detach().cpu().numpy()

    top_class_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[top_class_index]
    )

    top_class = LABELS[
        top_class_index
    ]

    return (
        probabilities,
        confidence,
        top_class,
        top_class_index
    )


# ============================================================
# GRAD-CAM
# ============================================================

def generate_gradcam(
    model,
    input_tensor,
    target_class
):
    """
    Generate Grad-CAM for the current AlexNet model.

    This version:
    - Does NOT modify model.py
    - Does NOT modify trained weights
    - Temporarily disables inplace ReLU
    - Hooks directly into the convolution output
    """

    activations = {}
    gradients = {}

    # ========================================================
    # TEMPORARILY DISABLE INPLACE RELU
    # ========================================================

    changed_relu_layers = []

    for module in model.modules():

        if isinstance(module, torch.nn.ReLU):

            if module.inplace:

                changed_relu_layers.append(module)
                module.inplace = False

    # ========================================================
    # FIND LAST CONVOLUTIONAL LAYER
    # ========================================================

    target_layer = None

    if hasattr(model, "backbone") and hasattr(
        model.backbone,
        "features"
    ):

        for layer in model.backbone.features:

            if isinstance(
                layer,
                torch.nn.Conv2d
            ):

                target_layer = layer

    if target_layer is None:

        for module in changed_relu_layers:
            module.inplace = True

        raise RuntimeError(
            "No Conv2d layer found in "
            "model.backbone.features."
        )

    # ========================================================
    # FORWARD HOOK
    # ========================================================

    def forward_hook(
        module,
        inputs,
        output
    ):

        # IMPORTANT:
        # Hook the ORIGINAL output directly.
        # Do NOT clone it.

        activations["value"] = output

        def gradient_hook(grad):

            gradients["value"] = grad

        output.register_hook(
            gradient_hook
        )

    # ========================================================
    # REGISTER HOOK
    # ========================================================

    forward_handle = target_layer.register_forward_hook(
        forward_hook
    )

    try:

        # ====================================================
        # ENABLE AUTOGRAD
        # ====================================================

        with torch.enable_grad():

            model.eval()

            model.zero_grad(
                set_to_none=True
            )

            # ------------------------------------------------
            # INPUT
            # ------------------------------------------------

            x = (
                input_tensor
                .clone()
                .detach()
                .to(DEVICE)
            )

            x.requires_grad_(True)

            # ------------------------------------------------
            # FORWARD
            # ------------------------------------------------

            output = model(x)

            # ------------------------------------------------
            # CHECK OUTPUT
            # ------------------------------------------------

            if "value" not in activations:

                raise RuntimeError(
                    "Grad-CAM activation "
                    "was not captured."
                )

            if output.ndim != 2:

                raise RuntimeError(
                    "Unexpected model output shape: "
                    f"{tuple(output.shape)}"
                )

            if (
                target_class < 0
                or target_class >= output.shape[1]
            ):

                raise RuntimeError(
                    f"Invalid target class: "
                    f"{target_class}. "
                    f"Model has "
                    f"{output.shape[1]} classes."
                )

            # ------------------------------------------------
            # TARGET CLASS SCORE
            # ------------------------------------------------

            target_score = output[
                :,
                target_class
            ].sum()

            # ------------------------------------------------
            # BACKWARD
            # ------------------------------------------------

            target_score.backward()

            # ------------------------------------------------
            # CHECK GRADIENT
            # ------------------------------------------------

            if "value" not in gradients:

                raise RuntimeError(
                    "Grad-CAM gradient "
                    "was not captured."
                )

            activation = activations["value"]
            gradient = gradients["value"]

            # =================================================
            # VALIDATE
            # =================================================

            if activation.ndim != 4:

                raise RuntimeError(
                    "Invalid activation shape: "
                    f"{tuple(activation.shape)}"
                )

            if gradient.ndim != 4:

                raise RuntimeError(
                    "Invalid gradient shape: "
                    f"{tuple(gradient.shape)}"
                )

            # =================================================
            # GRAD-CAM WEIGHTS
            # =================================================

            weights = gradient.mean(
                dim=(2, 3),
                keepdim=True
            )

            # =================================================
            # WEIGHTED ACTIVATIONS
            # =================================================

            cam = (
                weights * activation
            ).sum(
                dim=1
            )

            # =================================================
            # RELU
            # =================================================

            cam = F.relu(cam)

            # First image in batch

            cam = cam[0]

            # =================================================
            # DETACH + NUMPY
            # =================================================

            cam = (
                cam
                .detach()
                .cpu()
                .numpy()
            )

            # =================================================
            # NORMALIZE
            # =================================================

            cam_min = float(
                cam.min()
            )

            cam_max = float(
                cam.max()
            )

            if cam_max > cam_min:

                cam = (
                    cam - cam_min
                ) / (
                    cam_max - cam_min
                )

            else:

                cam = np.zeros_like(
                    cam,
                    dtype=np.float32
                )

            # =================================================
            # RESIZE
            # =================================================

            cam_image = Image.fromarray(
                np.uint8(
                    np.clip(
                        cam,
                        0,
                        1
                    ) * 255
                )
            )

            cam_image = cam_image.resize(
                (
                    IMG_SIZE,
                    IMG_SIZE
                ),
                Image.Resampling.BILINEAR
            )

            cam = np.asarray(
                cam_image,
                dtype=np.float32
            ) / 255.0

            # =================================================
            # FINAL CHECK
            # =================================================

            if not np.isfinite(cam).all():

                raise RuntimeError(
                    "Grad-CAM contains invalid values."
                )

            return cam

    finally:

        # ====================================================
        # REMOVE HOOK
        # ====================================================

        forward_handle.remove()

        # ====================================================
        # RESTORE RELU
        # ====================================================

        for module in changed_relu_layers:

            module.inplace = True


# ============================================================
# SAVE GRAD-CAM
# ============================================================

def save_gradcam(
    original_image,
    cam,
    output_path
):
    """
    Save Grad-CAM visualization.
    """

    # --------------------------------------------------------
    # Resize original image
    # --------------------------------------------------------

    original_image = original_image.resize(
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        Image.Resampling.BILINEAR
    )

    original = np.asarray(
        original_image,
        dtype=np.float32
    ) / 255.0

    # --------------------------------------------------------
    # CAM
    # --------------------------------------------------------

    cam = np.clip(
        cam,
        0,
        1
    )

    # --------------------------------------------------------
    # Create heatmap using PIL
    # --------------------------------------------------------

    heatmap = np.zeros(
        (
            IMG_SIZE,
            IMG_SIZE,
            3
        ),
        dtype=np.uint8
    )

    heatmap[:, :, 0] = np.uint8(
        cam * 255
    )

    heatmap[:, :, 1] = np.uint8(
        (1.0 - cam) * 120
    )

    heatmap[:, :, 2] = np.uint8(
        (1.0 - cam) * 255
    )

    heatmap = (
        heatmap.astype(
            np.float32
        ) / 255.0
    )

    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

    overlay = (
        0.55 * original
        +
        0.45 * heatmap
    )

    overlay = np.clip(
        overlay,
        0,
        1
    )

    overlay = np.uint8(
        overlay * 255
    )

    result = Image.fromarray(
        overlay
    )

    # --------------------------------------------------------
    # Create directory
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    result.save(
        output_path
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("        RETINAL FEDERATED MODEL PREDICTION")
    print("=" * 70)

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Model: AlexNet"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print(
        "\nClass mapping:"
    )

    for i, label in enumerate(LABELS):

        print(
            f"  {i} -> {label}"
        )

    # ========================================================
    # CHECK MODEL
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING GLOBAL MODEL"
    )

    print(
        "=" * 70
    )

    model = load_federated_model(
        MODEL_PATH
    )

    print(
        "\nGlobal model loaded successfully."
    )

    # ========================================================
    # LOAD IMAGE
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING TEST IMAGE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nImage:"
    )

    print(
        TEST_IMAGE
    )

    input_tensor, original_image = (
        load_image(
            TEST_IMAGE
        )
    )

    print(
        "\nImage loaded successfully."
    )

    print(
        f"Tensor shape: "
        f"{tuple(input_tensor.shape)}"
    )

    # ========================================================
    # PREDICTION
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "RUNNING PREDICTION"
    )

    print(
        "=" * 70
    )

    (
        probabilities,
        confidence,
        top_class,
        top_class_index
    ) = predict_image(
        model,
        input_tensor
    )

    # ========================================================
    # PRINT PREDICTIONS
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PREDICTIONS"
    )

    print(
        "-" * 70
    )

    for i, label in enumerate(LABELS):

        percentage = (
            probabilities[i]
            * 100.0
        )

        print(
            f"{label:<25}"
            f"{percentage:>7.2f}%"
        )

    print(
        "-" * 70
    )

    print(
        f"Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        f"Top class: "
        f"{top_class}"
    )

    print(
        f"Top class index: "
        f"{top_class_index}"
    )

    # ========================================================
    # GRAD-CAM
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERATING GRAD-CAM"
    )

    print(
        "=" * 70
    )

    try:

        cam = generate_gradcam(
            model,
            input_tensor,
            top_class_index
        )

        print(
            "\nGrad-CAM generated successfully."
        )

        print(
            f"Grad-CAM shape: "
            f"{cam.shape}"
        )

        # ----------------------------------------------------
        # SAVE GRAD-CAM
        # ----------------------------------------------------

        gradcam_path = os.path.join(
            PROJECT_ROOT,
            "results",
            "gradcam_prediction.png"
        )

        save_gradcam(
            original_image,
            cam,
            gradcam_path
        )

        print(
            "\nGrad-CAM saved to:"
        )

        print(
            gradcam_path
        )

    except Exception as e:

        print(
            "\nWARNING: Grad-CAM generation failed."
        )

        print(
            f"Reason: {e}"
        )

        print(
            "\nPrediction itself is still valid."
        )

        raise

    # ========================================================
    # FINAL TEST
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "PREDICTION TEST PASSED"
    )

    print(
        "=" * 70
    )

    print(
        "\n5-class prediction completed successfully."
    )

    print(
        "Classes:"
    )

    for label in LABELS:

        print(
            f"  - {label}"
        )

    print(
        "\nOutput files:"
    )

    print(
        f"  Global model:"
    )

    print(
        f"  {MODEL_PATH}"
    )

    print(
        f"\n  Grad-CAM:"
    )

    print(
        f"  {os.path.join(PROJECT_ROOT, 'results', 'gradcam_prediction.png')}"
    )

    print(
        "\n"
        + "=" * 70
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()