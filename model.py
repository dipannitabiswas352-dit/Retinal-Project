# ============================================================
# RETINAL FEDERATED LEARNING MODEL
# ============================================================

import torch
import torch.nn as nn
import torchvision.models as models

from config import (
    MODEL_NAME,
    IMG_SIZE,
    NUM_CLASSES,
    LABELS
)


# ============================================================
# MODEL VALIDATION
# ============================================================

if NUM_CLASSES != 5:
    raise ValueError(
        f"Expected 5 retinal condition classes, "
        f"but config.py contains NUM_CLASSES={NUM_CLASSES}"
    )

if len(LABELS) != NUM_CLASSES:
    raise ValueError(
        "NUM_CLASSES and LABELS length do not match."
    )


# ============================================================
# RETINAL CONDITION NAMES
# ============================================================

DISEASE_NAMES = LABELS


# ============================================================
# BACKBONE
# ============================================================

class Backbone(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # RESNET18
        # ----------------------------------------------------

        if MODEL_NAME.lower() == "resnet18":

            print("Using ResNet18")

            model = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT
            )

            # Remove final classification layer
            self.features = nn.Sequential(
                *list(model.children())[:-1]
            )

            self.feature_dim = 512

            # Fine-tune ResNet features
            for parameter in self.features.parameters():
                parameter.requires_grad = True

        # ----------------------------------------------------
        # ALEXNET
        # ----------------------------------------------------

        elif MODEL_NAME.lower() == "alexnet":

            print("Using AlexNet")

            model = models.alexnet(
                weights=models.AlexNet_Weights.DEFAULT
            )

            # Convolutional feature extractor
            self.features = model.features

            # AlexNet adaptive pooling
            self.avgpool = model.avgpool

            # ------------------------------------------------
            # FREEZE CONVOLUTIONAL FEATURES
            # ------------------------------------------------

            for parameter in self.features.parameters():
                parameter.requires_grad = False

            # ------------------------------------------------
            # TRAINABLE FEATURE PROJECTION
            # ------------------------------------------------

            self.classifier = nn.Sequential(

                nn.Flatten(),

                nn.Linear(
                    256 * 6 * 6,
                    256
                ),

                nn.ReLU(inplace=True),

                nn.Dropout(0.3),

                nn.Linear(
                    256,
                    512
                ),

                nn.ReLU(inplace=True),

                nn.Dropout(0.3)
            )

            self.feature_dim = 512

        else:

            raise ValueError(
                f"Unsupported MODEL_NAME: {MODEL_NAME}"
            )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        # ----------------------------------------------------
        # INPUT VALIDATION
        # ----------------------------------------------------

        if x.ndim != 4:

            raise ValueError(
                f"Expected 4D input [B,C,H,W], "
                f"got shape {tuple(x.shape)}"
            )

        if x.shape[1] != 3:

            raise ValueError(
                f"Expected 3 input channels, "
                f"got {x.shape[1]}"
            )

        if (
            x.shape[2] != IMG_SIZE
            or x.shape[3] != IMG_SIZE
        ):

            raise ValueError(
                f"Expected image size "
                f"{IMG_SIZE}x{IMG_SIZE}, "
                f"got {x.shape[2]}x{x.shape[3]}"
            )

        # ----------------------------------------------------
        # ALEXNET
        # ----------------------------------------------------

        if MODEL_NAME.lower() == "alexnet":

            x = self.features(x)

            x = self.avgpool(x)

            x = torch.flatten(
                x,
                start_dim=1
            )

            x = self.classifier(x)

            return x

        # ----------------------------------------------------
        # RESNET18
        # ----------------------------------------------------

        elif MODEL_NAME.lower() == "resnet18":

            x = self.features(x)

            x = torch.flatten(
                x,
                start_dim=1
            )

            return x

        raise RuntimeError(
            "Unsupported backbone during forward pass."
        )


# ============================================================
# CLASSIFICATION HEAD
# ============================================================

class ClassificationHead(nn.Module):

    def __init__(
        self,
        input_features,
        num_classes=NUM_CLASSES
    ):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(
                input_features,
                512
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.5),

            nn.Linear(
                512,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                num_classes
            )
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        return self.net(x)


# ============================================================
# COMPLETE FEDERATED MODEL
# ============================================================

class Net(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = Backbone()

        self.head = ClassificationHead(
            self.backbone.feature_dim,
            NUM_CLASSES
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        features = self.backbone(x)

        output = self.head(features)

        return output


    # ========================================================
    # GET TRAINABLE FEDERATED PARAMETERS
    # ========================================================

    def get_shared_parameters(self):

        return [
            parameter.detach().cpu().numpy()
            for parameter in self.parameters()
            if parameter.requires_grad
        ]


    # ========================================================
    # SET TRAINABLE FEDERATED PARAMETERS
    # ========================================================

    def set_shared_parameters(
        self,
        parameters
    ):

        trainable_parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        if len(parameters) != len(trainable_parameters):

            raise ValueError(
                "\nFederated parameter count mismatch.\n"
                f"Received: {len(parameters)}\n"
                f"Expected: {len(trainable_parameters)}"
            )

        for parameter, new_value in zip(
            trainable_parameters,
            parameters
        ):

            tensor = torch.tensor(
                new_value,
                dtype=parameter.dtype,
                device=parameter.device
            )

            if parameter.shape != tensor.shape:

                raise ValueError(
                    "\nFederated parameter shape mismatch.\n"
                    f"Expected: {tuple(parameter.shape)}\n"
                    f"Received: {tuple(tensor.shape)}"
                )

            parameter.data.copy_(tensor)


# ============================================================
# MODEL INFORMATION
# ============================================================

def print_model_information():

    print("\n")
    print("=" * 60)
    print("RETINAL FEDERATED MODEL INFORMATION")
    print("=" * 60)

    print(
        f"Model name       : {MODEL_NAME}"
    )

    print(
        f"Image size       : {IMG_SIZE}x{IMG_SIZE}"
    )

    print(
        f"Number of classes: {NUM_CLASSES}"
    )

    print(
        f"Feature dimension: 512"
    )

    print("\nClasses:")

    for index, label in enumerate(
        DISEASE_NAMES
    ):

        print(
            f"  {index}: {label}"
        )

    model = Net()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        f"\nTrainable tensors : "
        f"{len(trainable_parameters)}"
    )

    print("\nTrainable tensor shapes:")

    for index, parameter in enumerate(
        trainable_parameters
    ):

        print(
            f"  Tensor {index + 1}: "
            f"{tuple(parameter.shape)}"
        )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"\nTotal parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Trainable parameters : "
        f"{trainable_count:,}"
    )

    print("=" * 60)


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("TESTING RETINAL FEDERATED MODEL")
    print("=" * 60)

    # --------------------------------------------------------
    # PRINT INFORMATION
    # --------------------------------------------------------

    print_model_information()

    # --------------------------------------------------------
    # CREATE MODEL
    # --------------------------------------------------------

    print("\nCreating model...")

    model = Net()

    print(
        "Model created successfully."
    )

    # --------------------------------------------------------
    # RANDOM TEST INPUT
    # --------------------------------------------------------

    test_input = torch.randn(
        2,
        3,
        IMG_SIZE,
        IMG_SIZE
    )

    print(
        f"\nTest input shape: "
        f"{tuple(test_input.shape)}"
    )

    # --------------------------------------------------------
    # FORWARD PASS
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(
            test_input
        )

    print(
        f"Output shape: "
        f"{tuple(output.shape)}"
    )

    # --------------------------------------------------------
    # VERIFY OUTPUT
    # --------------------------------------------------------

    expected_shape = (
        2,
        NUM_CLASSES
    )

    if tuple(output.shape) != expected_shape:

        raise RuntimeError(
            "\nMODEL TEST FAILED\n"
            f"Expected output shape: "
            f"{expected_shape}\n"
            f"Actual output shape: "
            f"{tuple(output.shape)}"
        )

    print(
        "\n5-class output verification: PASSED"
    )

    # --------------------------------------------------------
    # FEDERATED PARAMETER TEST
    # --------------------------------------------------------

    federated_parameters = (
        model.get_shared_parameters()
    )

    print(
        f"\nFederated tensors: "
        f"{len(federated_parameters)}"
    )

    # --------------------------------------------------------
    # TEST PARAMETER ROUND TRIP
    # --------------------------------------------------------

    model.set_shared_parameters(
        federated_parameters
    )

    print(
        "Federated parameter "
        "round-trip test: PASSED"
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("MODEL TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(
        "\nFinal classes:"
    )

    for index, label in enumerate(
        DISEASE_NAMES
    ):

        print(
            f"  {index}: {label}"
        )

    print("\nExpected output:")

    print(
        "  [batch_size, 5]"
    )

    print("=" * 60)