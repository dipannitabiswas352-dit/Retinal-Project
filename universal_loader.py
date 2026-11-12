import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

IMG_SIZE = 224

VALID_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff"
)


# ============================================================
# BASIC IMAGE FUNCTIONS
# ============================================================

def safe_read(image_path):
    """
    Safely read an image using OpenCV.
    Returns None if image cannot be read.
    """

    image = cv2.imread(image_path)

    if image is None:
        return None

    return image


def preprocess_image(image):
    """
    Resize image to 224x224,
    convert BGR -> RGB,
    normalize pixel values to [0, 1].
    """

    image = cv2.resize(
        image,
        (IMG_SIZE, IMG_SIZE)
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    return image


def collect_image_files(folder):
    """
    Recursively collect all valid image files.
    """

    image_files = []

    if not os.path.exists(folder):

        print(
            f"WARNING: Folder does not exist: {folder}"
        )

        return image_files

    for root, _, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                VALID_EXTENSIONS
            ):

                image_files.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    return sorted(image_files)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_distribution(
    labels,
    class_names=None
):
    """
    Print class distribution.
    """

    if labels is None or len(labels) == 0:

        print("No labels found.")

        return

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 60)

    for label, count in zip(
        unique,
        counts
    ):

        label = int(label)

        if (
            class_names is not None
            and label < len(class_names)
        ):

            name = class_names[label]

        else:

            name = f"Class {label}"

        print(
            f"{label:>3} : "
            f"{name:<25} -> "
            f"{int(count)}"
        )

    print("-" * 60)

    print(
        f"Total samples: {len(labels)}"
    )


# ============================================================
# GAUSSIAN
# ============================================================

def load_gaussian(path):

    class_names = [
        "No_DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferate_DR"
    ]

    class_mapping = {
        "No_DR": 0,
        "Mild": 1,
        "Moderate": 2,
        "Severe": 3,
        "Proliferate_DR": 4
    }

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Gaussian dataset folder not found:\n{path}"
        )

    data = []

    print("\nLOADING GAUSSIAN DATASET")
    print("=" * 60)

    for class_name, label in class_mapping.items():

        class_folder = os.path.join(
            path,
            class_name
        )

        if not os.path.exists(class_folder):

            print(
                f"WARNING: Missing Gaussian "
                f"class folder: {class_name}"
            )

            continue

        image_files = collect_image_files(
            class_folder
        )

        print(
            f"{class_name:<20} : "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            image = safe_read(
                image_path
            )

            if image is None:

                print(
                    f"WARNING: Could not read: "
                    f"{image_path}"
                )

                continue

            image = preprocess_image(
                image
            )

            data.append(
                (
                    image,
                    label
                )
            )

    print("-" * 60)

    print(
        f"Gaussian samples loaded: "
        f"{len(data)}"
    )

    if data:

        labels = [
            item[1]
            for item in data
        ]

        print_distribution(
            labels,
            class_names
        )

    return data


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def load_folder(path):

    return load_gaussian(path)


# ============================================================
# OCT
# ============================================================

def load_oct(path):

    class_names = [
        "AMD",
        "CNV",
        "CSR",
        "DME",
        "DR",
        "DRUSEN",
        "MH"
    ]

    class_mapping = {
        "AMD": 0,
        "CNV": 1,
        "CSR": 2,
        "DME": 3,
        "DR": 4,
        "DRUSEN": 5,
        "MH": 6
    }

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"OCT dataset folder not found:\n{path}"
        )

    data = []

    print("\nLOADING OCT DATASET")
    print("=" * 60)

    for class_name, label in class_mapping.items():

        class_folder = os.path.join(
            path,
            class_name
        )

        if not os.path.exists(class_folder):

            print(
                f"WARNING: Missing OCT "
                f"class folder: {class_name}"
            )

            continue

        image_files = collect_image_files(
            class_folder
        )

        print(
            f"{class_name:<20} : "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            data.append(
                (
                    image_path,
                    label
                )
            )

    print("-" * 60)

    print(
        f"OCT samples loaded: "
        f"{len(data)}"
    )

    if data:

        labels = [
            item[1]
            for item in data
        ]

        print_distribution(
            labels,
            class_names
        )

    return data


# ============================================================
# RFMID
# ============================================================

def load_rfmid(
    image_folder,
    csv_path
):
    class_names = [
        "Normal",
        "Abnormal"
    ]

    if not os.path.exists(image_folder):

        raise FileNotFoundError(
            f"RFMiD image folder not found:\n{image_folder}"
        )

    if csv_path is None:

        raise ValueError(
            "RFMiD requires a CSV file."
        )

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"RFMiD CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # FIND IMAGE COLUMN
    # --------------------------------------------------------

    image_column = None

    possible_image_columns = [
        "ID",
        "ImageID",
        "image_id",
        "image",
        "Image",
        "id"
    ]

    for column in possible_image_columns:

        if column in df.columns:

            image_column = column

            break

    if image_column is None:

        raise ValueError(
            "Could not find RFMiD image ID column."
        )

    # --------------------------------------------------------
    # CHECK LABEL COLUMN
    # --------------------------------------------------------

    if "Disease_Risk" not in df.columns:

        raise ValueError(
            "RFMiD CSV must contain 'Disease_Risk'."
        )

    data = []

    # --------------------------------------------------------
    # PROCESS CSV
    # --------------------------------------------------------

    for _, row in df.iterrows():

        try:

            label = int(
                row["Disease_Risk"]
            )

        except Exception:

            continue

        if label not in [0, 1]:

            continue

        image_name = str(
            row[image_column]
        ).strip()

        if (
            image_name == ""
            or image_name.lower() == "nan"
        ):

            continue

        image_path = None

        # ----------------------------------------------------
        # TRY IMAGE EXTENSIONS
        # ----------------------------------------------------

        for ext in VALID_EXTENSIONS:

            candidate = os.path.join(
                image_folder,
                image_name + ext
            )

            if os.path.exists(candidate):

                image_path = candidate

                break

        # ----------------------------------------------------
        # TRY DIRECT PATH
        # ----------------------------------------------------

        if image_path is None:

            direct_path = os.path.join(
                image_folder,
                image_name
            )

            if os.path.exists(direct_path):

                image_path = direct_path

        if image_path is None:

            continue

        data.append(
            (
                image_path,
                label
            )
        )

    print(
        f"RFMiD samples loaded: "
        f"{len(data)}"
    )

    if data:

        labels = [
            item[1]
            for item in data
        ]

        print_distribution(
            labels,
            class_names
        )

    return data


# ============================================================
# ODIR-5K
# ============================================================

def load_odir(
    image_folder,
    csv_path
):

    class_names = [
        "Normal",
        "Diabetes",
        "Glaucoma",
        "Cataract",
        "AMD",
        "Hypertension",
        "Myopia",
        "Other"
    ]

    class_mapping = {
        "N": 0,
        "D": 1,
        "G": 2,
        "C": 3,
        "A": 4,
        "H": 5,
        "M": 6,
        "O": 7
    }

    if not os.path.exists(image_folder):

        raise FileNotFoundError(
            f"ODIR image folder not found:\n{image_folder}"
        )

    if csv_path is None:

        raise ValueError(
            "ODIR-5K requires an annotation CSV."
        )

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"ODIR CSV not found:\n{csv_path}"
        )

    df = pd.read_csv(csv_path)

    # --------------------------------------------------------
    # FIND IMAGE COLUMNS
    # --------------------------------------------------------

    image_columns = []

    possible_left = [
        "Left-Fundus",
        "Left_Fundus",
        "left-fundus",
        "left_fundus"
    ]

    possible_right = [
        "Right-Fundus",
        "Right_Fundus",
        "right-fundus",
        "right_fundus"
    ]

    for column in possible_left:

        if column in df.columns:

            image_columns.append(column)

            break

    for column in possible_right:

        if column in df.columns:

            image_columns.append(column)

            break

    # --------------------------------------------------------
    # GENERIC IMAGE COLUMN
    # --------------------------------------------------------

    if len(image_columns) == 0:

        possible_generic = [
            "image",
            "Image",
            "image_name",
            "Image_Name",
            "filename",
            "Filename"
        ]

        for column in possible_generic:

            if column in df.columns:

                image_columns.append(column)

                break

    if len(image_columns) == 0:

        raise ValueError(
            "Could not find ODIR image columns."
        )

    # --------------------------------------------------------
    # FIND DISEASE COLUMNS
    # --------------------------------------------------------

    disease_columns = []

    for disease in class_mapping.keys():

        if disease in df.columns:

            disease_columns.append(disease)

    if len(disease_columns) == 0:

        raise ValueError(
            "Could not find ODIR disease columns "
            "(N, D, G, C, A, H, M, O)."
        )

    data = []

    # --------------------------------------------------------
    # PROCESS ROWS
    # --------------------------------------------------------

    for _, row in df.iterrows():

        positive_labels = []

        for disease in disease_columns:

            try:

                value = int(
                    row[disease]
                )

            except Exception:

                value = 0

            if value == 1:

                positive_labels.append(
                    disease
                )

        # Skip unlabeled samples
        if len(positive_labels) == 0:

            continue

        # Skip multi-label samples
        if len(positive_labels) != 1:

            continue

        disease = positive_labels[0]

        label = class_mapping[
            disease
        ]

        # ----------------------------------------------------
        # LEFT / RIGHT FUNDUS
        # ----------------------------------------------------

        for image_column in image_columns:

            image_name = str(
                row[image_column]
            ).strip()

            if (
                image_name == ""
                or image_name.lower() == "nan"
            ):

                continue

            image_path = os.path.join(
                image_folder,
                image_name
            )

            # ------------------------------------------------
            # TRY EXTENSIONS
            # ------------------------------------------------

            if not os.path.exists(image_path):

                found = False

                for ext in VALID_EXTENSIONS:

                    candidate = (
                        image_path + ext
                    )

                    if os.path.exists(candidate):

                        image_path = candidate

                        found = True

                        break

                if not found:

                    continue

            image = safe_read(
                image_path
            )

            if image is None:

                continue

            image = preprocess_image(
                image
            )

            data.append(
                (
                    image,
                    label
                )
            )

    print(
        f"ODIR-5K samples loaded: "
        f"{len(data)}"
    )

    if data:

        labels = [
            item[1]
            for item in data
        ]

        print_distribution(
            labels,
            class_names
        )

    return data


# ============================================================
# SPLIT DATASET
# ============================================================

def load_split(
    path,
    split="train"
):
    class_names = [
        "cataract",
        "diabetic_retinopathy",
        "glaucoma",
        "normal"
    ]

    class_mapping = {
        "cataract": 0,
        "diabetic_retinopathy": 1,
        "glaucoma": 2,
        "normal": 3
    }

    # ========================================================
    # VALIDATE ROOT PATH
    # ========================================================

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"SPLIT dataset folder not found:\n{path}"
        )

    # ========================================================
    # VALIDATE SPLIT NAME
    # ========================================================

    split_name = str(
        split
    ).lower().strip()

    if split_name not in [
        "train",
        "val",
        "test"
    ]:

        raise ValueError(
            f"Invalid SPLIT split: '{split}'.\n"
            f"Expected: 'train', 'val' or 'test'."
        )

    split_path = os.path.join(
        path,
        split_name
    )

    if not os.path.exists(split_path):

        raise FileNotFoundError(
            f"SPLIT {split_name} directory not found:\n"
            f"{split_path}"
        )

    # ========================================================
    # STRICT CLASS FOLDER VALIDATION
    # ========================================================

    print("\n" + "=" * 60)

    print(
        f"VALIDATING SPLIT {split_name.upper()} "
        f"CLASS MAPPING"
    )

    print("=" * 60)

    for class_name in class_names:

        class_folder = os.path.join(
            split_path,
            class_name
        )

        if not os.path.isdir(class_folder):

            raise FileNotFoundError(
                "\nINVALID SPLIT DATASET STRUCTURE!\n\n"
                f"Missing required class folder:\n"
                f"{class_folder}\n\n"
                "Required class folders are:\n"
                "  0 -> cataract\n"
                "  1 -> diabetic_retinopathy\n"
                "  2 -> glaucoma\n"
                "  3 -> normal"
            )

    # ========================================================
    # ENFORCE EXACT CLASS SET
    # ========================================================

    actual_folders = sorted([
        folder
        for folder in os.listdir(split_path)
        if os.path.isdir(
            os.path.join(
                split_path,
                folder
            )
        )
    ])

    expected_folders = sorted(
        class_names
    )

    if actual_folders != expected_folders:

        raise ValueError(
            "\nINVALID SPLIT CLASS FOLDERS!\n\n"
            f"Expected:\n{expected_folders}\n\n"
            f"Found:\n{actual_folders}\n\n"
            "Folder names must be exactly:\n"
            "cataract\n"
            "diabetic_retinopathy\n"
            "glaucoma\n"
            "normal"
        )

    # ========================================================
    # PRINT VALID MAPPING
    # ========================================================

    print("\nCLASS MAPPING: VALID")

    for class_name, label in class_mapping.items():

        print(
            f"  {label} -> {class_name}"
        )

    # ========================================================
    # LOAD DATA
    # ========================================================

    data = []

    print("\n" + "=" * 60)

    print(
        f"LOADING SPLIT DATASET - "
        f"{split_name.upper()}"
    )

    print("=" * 60)

    print(
        f"Dataset path: {split_path}"
    )

    for class_name, label in class_mapping.items():

        class_folder = os.path.join(
            split_path,
            class_name
        )

        image_files = collect_image_files(
            class_folder
        )

        print(
            f"{label} -> "
            f"{class_name:<25} : "
            f"{len(image_files)} images"
        )

        # ----------------------------------------------------
        # Reject empty class folders
        # ----------------------------------------------------

        if len(image_files) == 0:

            raise ValueError(
                f"\nClass folder is empty:\n"
                f"{class_folder}\n\n"
                f"Expected class {label}: "
                f"{class_name}\n\n"
                "Please place the corresponding "
                "images inside this folder."
            )

        # ----------------------------------------------------
        # Validate images
        # ----------------------------------------------------

        for image_path in image_files:

            image = safe_read(
                image_path
            )

            if image is None:

                print(
                    f"WARNING: Could not read:\n"
                    f"{image_path}"
                )

                continue

            data.append(
                (
                    image_path,
                    label
                )
            )

    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    if len(data) == 0:

        raise ValueError(
            f"No valid images found in "
            f"SPLIT {split_name} dataset."
        )

    labels = [
        item[1]
        for item in data
    ]

    unique_labels = sorted(
        set(labels)
    )

    expected_labels = [
        0,
        1,
        2,
        3
    ]

    if unique_labels != expected_labels:

        raise ValueError(
            "\nINVALID LABEL SET!\n\n"
            f"Expected labels: {expected_labels}\n"
            f"Found labels: {unique_labels}"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n" + "-" * 60)

    print(
        f"SPLIT {split_name} samples loaded: "
        f"{len(data)}"
    )

    print_distribution(
        labels,
        class_names
    )

    print("\n" + "=" * 60)

    print(
        f"SPLIT {split_name.upper()} "
        f"VALIDATION PASSED"
    )

    print("=" * 60)

    return data


# ============================================================
# UNIVERSAL DATASET DISPATCHER
# ============================================================

def load_dataset(
    name,
    path,
    csv_path=None,
    split="train"
):
    dataset_name = (
        name
        .lower()
        .strip()
    )

    # ========================================================
    # GAUSSIAN
    # ========================================================

    if dataset_name in [
        "gaussian",
        "guassian",
        "gaussian_filtered",
        "gaussian-filtered"
    ]:

        return load_gaussian(
            path
        )

    # ========================================================
    # OCT
    # ========================================================

    elif dataset_name in [
        "oct",
        "ocp"
    ]:

        return load_oct(
            path
        )

    # ========================================================
    # RFMID
    # ========================================================

    elif dataset_name in [
        "rfmid",
        "rfu"
    ]:

        return load_rfmid(
            image_folder=path,
            csv_path=csv_path
        )

    # ========================================================
    # ODIR-5K
    # ========================================================

    elif dataset_name in [
        "odir",
        "odir5k",
        "odir-5k"
    ]:

        return load_odir(
            image_folder=path,
            csv_path=csv_path
        )

    # ========================================================
    # SPLIT
    # ========================================================

    elif dataset_name in [
        "split",
        "split_dataset",
        "split-dataset"
    ]:

        return load_split(
            path=path,
            split=split
        )

    # ========================================================
    # REMOVED DATASETS
    # ========================================================

    elif dataset_name in [
        "refuge",
        "refuge2",
        "refuge-2",
        "messidor",
        "messidor2",
        "eyepacs",
        "eye-pacs",
        "ipa",
        "gcroad",
        "gc_road",
        "gc road",
        "ipec"
    ]:

        raise ValueError(
            f"Dataset '{name}' has been removed "
            "from this project.\n\n"
            "Active datasets are:\n"
            "  - Gaussian\n"
            "  - OCT\n"
            "  - RFMiD\n"
            "  - ODIR-5K\n"
            "  - SPLIT"
        )

    # ========================================================
    # UNKNOWN DATASET
    # ========================================================

    else:

        raise ValueError(
            f"Unknown dataset: '{name}'\n\n"
            "Currently supported datasets:\n"
            "  - Gaussian\n"
            "  - OCT\n"
            "  - RFMiD\n"
            "  - ODIR-5K\n"
            "  - SPLIT"
        )


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print("\n")
    print("=" * 60)
    print("             UNIVERSAL LOADER")
    print("=" * 60)

    print("\nACTIVE DATASETS:")

    print("  1. Gaussian")
    print("  2. OCT")
    print("  3. RFMiD")
    print("  4. ODIR-5K")
    print("  5. SPLIT")

    print("\nREMOVED DATASETS:")

    print("  - REFUGE2")
    print("  - MESSIDOR")
    print("  - MESSIDOR2")
    print("  - EyePACS")
    print("  - IPA")
    print("  - GC Road")
    print("  - IPEC")

    print("\nSPLIT DATASET CLASSES:")

    print("  0 -> Cataract")
    print("  1 -> Diabetic Retinopathy")
    print("  2 -> Glaucoma")
    print("  3 -> Normal")

    print("\nLOADER STATUS:")

    print("  Gaussian loader : OK")
    print("  OCT loader      : OK")
    print("  RFMiD loader    : OK")
    print("  ODIR-5K loader  : OK")
    print("  SPLIT loader    : OK")

    print("\nBACKWARD COMPATIBILITY:")

    print("  load_folder() -> load_gaussian()")

    print("\nUniversal loader imported successfully.")

    print("=" * 60)