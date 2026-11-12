import re

CLASS_NAMES = [
    "Diabetic Retinopathy",
    "Glaucoma",
    "AMD",
    "Cataract",
    "Normal"
]

LABELS = CLASS_NAMES

NUM_CLASSES = 5


CLASS_TO_INDEX = {
    "Diabetic Retinopathy": 0,
    "DR": 0,
    "diabetic retinopathy": 0,
    "dr": 0,

    "Glaucoma": 1,
    "glaucoma": 1,
    "Gla": 1,

    "AMD": 2,
    "amd": 2,
    "ARMD": 2,
    "armd": 2,
    "Age-related Macular Degeneration": 2,
    "Age Related Macular Degeneration": 2,
    "age related macular degeneration": 2,

    "Cataract": 3,
    "cataract": 3,

    "Normal": 4,
    "normal": 4,
    "No DR": 4,
    "no dr": 4,
    "No Disease": 4,
    "no disease": 4
}


# ============================================================
# UNKNOWN LABEL
# ============================================================

UNKNOWN = -1


# ============================================================
# NORMALIZE LABEL TEXT
# ============================================================

def normalize_label(value):
    """
    Normalize a dataset label.

    Examples:
        "Diabetic_Retinopathy" -> "diabetic retinopathy"
        "Age-related Macular Degeneration"
            -> "age related macular degeneration"
    """

    if value is None:
        return ""

    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NORMALIZE CLASS NAME
# ============================================================

def get_class_index(label):
    """
    Convert a dataset label into the global class index.

    Returns:
        0 = Diabetic Retinopathy
        1 = Glaucoma
        2 = AMD
        3 = Cataract
        4 = Normal
       -1 = Unknown / unsupported
    """

    if label is None:
        return UNKNOWN

    # --------------------------------------------------------
    # Numeric class index
    # --------------------------------------------------------

    if isinstance(label, int):

        if 0 <= label < NUM_CLASSES:
            return label

        return UNKNOWN

    # --------------------------------------------------------
    # Numeric string
    # --------------------------------------------------------

    text = normalize_label(label)

    if text.isdigit():

        numeric = int(text)

        if 0 <= numeric < NUM_CLASSES:
            return numeric

    # --------------------------------------------------------
    # Exact dictionary match
    # --------------------------------------------------------

    if text in CLASS_TO_INDEX:
        return CLASS_TO_INDEX[text]

    # --------------------------------------------------------
    # DIABETIC RETINOPATHY
    # --------------------------------------------------------

    dr_patterns = [
        "diabetic retinopathy",
        "mild dr",
        "moderate dr",
        "severe dr",
        "proliferative dr",
        "proliferative diabetic retinopathy",
        "non proliferative diabetic retinopathy",
        "npdr",
        "pdr"
    ]

    for pattern in dr_patterns:

        if pattern in text:
            return 0

    # --------------------------------------------------------
    # GLAUCOMA
    # --------------------------------------------------------

    glaucoma_patterns = [
        "glaucoma",
        "glaucomatous",
        "glaucoma suspect",
        "optic disc glaucoma",
        "optic nerve glaucoma"
    ]

    for pattern in glaucoma_patterns:

        if pattern in text:
            return 1

    # --------------------------------------------------------
    # AMD
    # --------------------------------------------------------

    amd_patterns = [
        "age related macular degeneration",
        "age related macular degeneration",
        "macular degeneration",
        "wet amd",
        "dry amd",
        "neovascular amd",
        "non neovascular amd"
    ]

    for pattern in amd_patterns:

        if pattern in text:
            return 2

    # --------------------------------------------------------
    # CATARACT
    # --------------------------------------------------------

    cataract_patterns = [
        "cataract",
        "cataracts",
        "lens opacity"
    ]

    for pattern in cataract_patterns:

        if pattern in text:
            return 3

    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    normal_patterns = [
        "normal",
        "no disease",
        "no disease detected",
        "healthy",
        "healthy eye",
        "no dr",
        "no diabetic retinopathy"
    ]

    for pattern in normal_patterns:

        if text == pattern:
            return 4

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    return UNKNOWN


# ============================================================
# CLASS NAME FROM INDEX
# ============================================================

def get_class_name(index):
    """
    Convert global class index into class name.
    """

    try:
        index = int(index)
    except (ValueError, TypeError):

        raise ValueError(
            f"Invalid class index: {index}"
        )

    if index < 0 or index >= NUM_CLASSES:

        raise ValueError(
            f"Class index must be between "
            f"0 and {NUM_CLASSES - 1}, "
            f"got {index}"
        )

    return CLASS_NAMES[index]


# ============================================================
# VALIDATE CLASS INDEX
# ============================================================

def validate_class_index(index):
    """
    Check whether a class index is valid.
    """

    try:
        index = int(index)
    except (ValueError, TypeError):

        return False

    return 0 <= index < NUM_CLASSES


# ============================================================
# DATASET-SPECIFIC LABEL MAPPINGS
# ============================================================


# ------------------------------------------------------------
# GAUSSIAN / EYE-PACS STYLE DR LABEL
# ------------------------------------------------------------

def map_dr_grade(label):
    """
    Map diabetic-retinopathy severity into the global
    5-class system.

    Gaussian / EyePACS style:

        0 = No DR
        1 = Mild
        2 = Moderate
        3 = Severe
        4 = Proliferative DR

    Mapping:

        0 -> Normal
        1-4 -> Diabetic Retinopathy

    Returns:
        global class index
    """

    try:

        numeric = int(label)

    except (ValueError, TypeError):

        text = normalize_label(label)

        if text in [
            "no dr",
            "no diabetic retinopathy",
            "normal"
        ]:

            return 4

        if any(
            keyword in text
            for keyword in [
                "mild",
                "moderate",
                "severe",
                "proliferative",
                "retinopathy"
            ]
        ):

            return 0

        return UNKNOWN

    if numeric == 0:
        return 4

    if numeric in [1, 2, 3, 4]:
        return 0

    return UNKNOWN


# ============================================================
# GAUSSIAN FOLDER LABEL MAPPING
# ============================================================

def map_gaussian_label(label):
    """
    Map Gaussian dataset folder names.

    Expected folders:

        No_DR
        Mild
        Moderate
        Severe
        Proliferate_DR

    Returns:
        0 = DR
        4 = Normal
       -1 = Unknown
    """

    text = normalize_label(label)

    if text in [
        "no dr",
        "normal"
    ]:
        return 4

    if text in [
        "mild",
        "moderate",
        "severe",
        "proliferate dr",
        "proliferative dr"
    ]:
        return 0

    return map_dr_grade(label)


# ============================================================
# OCT LABEL MAPPING
# ============================================================

def map_oct_label(label):
    """
    Map OCT dataset labels into supported global classes.

    Supported:

        AMD
        DR

    Unsupported OCT classes remain UNKNOWN.

    This is intentional. We do NOT convert unrelated
    retinal conditions into DR/AMD.
    """

    text = normalize_label(label)

    # AMD-related OCT classes
    if text in [
        "amd",
        "armd",
        "age related macular degeneration"
    ]:
        return 2

    # Some OCT datasets use CNV as an AMD-related class.
    if text == "cnv":
        return 2

    # Diabetic retinopathy
    if text in [
        "dr",
        "dme",
        "diabetic retinopathy"
    ]:
        return 0

    # Unsupported OCT categories:
    #
    # CSR
    # DRUSEN
    # MH
    #
    # are NOT automatically converted.

    return UNKNOWN


# ============================================================
# ODIR-5K LABEL MAPPING
# ============================================================

def map_odir_label(label):
    """
    Map ODIR-5K folder labels.

    Supported ODIR classes:

        ageDegeneration -> AMD
        cataract       -> Cataract
        diabetes       -> DR
        glaucoma       -> Glaucoma
        normal         -> Normal

    Unsupported:

        hypertension
        myopia
    """

    text = normalize_label(label)

    # AMD
    if text in [
        "agedegeneration",
        "age degeneration",
        "age related macular degeneration",
        "amd"
    ]:
        return 2

    # Cataract
    if text == "cataract":
        return 3

    # Diabetic retinopathy
    if text in [
        "diabetes",
        "diabetic retinopathy",
        "dr"
    ]:
        return 0

    # Glaucoma
    if text == "glaucoma":
        return 1

    # Normal
    if text == "normal":
        return 4

    # Do NOT map hypertension/myopia
    return UNKNOWN


# ============================================================
# SPLIT DATASET LABEL MAPPING
# ============================================================

def map_split_label(label):
    """
    Map the new split dataset.

    Expected folders:

        cataract
        diabetic retinopathy
        glaucoma
        normal
    """

    text = normalize_label(label)

    if text == "cataract":
        return 3

    if text in [
        "diabetic retinopathy",
        "dr"
    ]:
        return 0

    if text == "glaucoma":
        return 1

    if text == "normal":
        return 4

    return UNKNOWN


# ============================================================
# RFMiD LABEL MAPPING
# ============================================================

def map_rfmID_label(label):
    """
    Map an RFMiD disease label.

    This function only maps labels that explicitly identify
    one of our five target classes.

    A generic 'abnormal' or Disease_Risk=1 MUST NOT
    automatically become DR.
    """

    text = normalize_label(label)

    # Normal
    if text in [
        "normal",
        "no disease",
        "healthy"
    ]:
        return 4

    # DR
    if "diabetic retinopathy" in text:
        return 0

    if text in [
        "dr",
        "npdr",
        "pdr"
    ]:
        return 0

    # Glaucoma
    if "glaucoma" in text:
        return 1

    # AMD
    if (
        "amd" in text
        or
        "armd" in text
        or
        "macular degeneration" in text
    ):
        return 2

    # Cataract
    if "cataract" in text:
        return 3

    # Generic abnormal = UNKNOWN
    if text in [
        "abnormal",
        "disease",
        "disease risk",
        "positive"
    ]:
        return UNKNOWN

    return UNKNOWN


# ============================================================
# GENERIC DATASET LABEL MAPPER
# ============================================================

def map_dataset_label(
    dataset,
    label
):
    """
    Main dataset-independent mapping function.

    Example:

        map_dataset_label("gaussian", "Moderate")
            -> 0

        map_dataset_label("odir", "glaucoma")
            -> 1

        map_dataset_label("split", "normal")
            -> 4
    """

    dataset_name = normalize_label(dataset)

    if dataset_name in [
        "gaussian",
        "gaussian filtered images",
        "gaussian_filtered_images"
    ]:
        return map_gaussian_label(label)

    if dataset_name in [
        "oct",
        "retinaloct",
        "retinal oct",
        "retinaloct dataset"
    ]:
        return map_oct_label(label)

    if dataset_name in [
        "odir",
        "odir 5k",
        "odir 5k final",
        "odir5k",
        "odir5k final"
    ]:
        return map_odir_label(label)

    if dataset_name in [
        "rfmid",
        "rfmid dataset",
        "rfmid_dataset"
    ]:
        return map_rfmID_label(label)

    if dataset_name in [
        "split",
        "split dataset"
    ]:
        return map_split_label(label)

    return get_class_index(label)


# ============================================================
# CHECK WHETHER LABEL IS SUPPORTED
# ============================================================

def is_supported_label(
    dataset,
    label
):
    """
    Return True when a label can safely be mapped to one
    of the five global classes.
    """

    return (
        map_dataset_label(
            dataset,
            label
        )
        != UNKNOWN
    )


# ============================================================
# PRINT LABEL INFORMATION
# ============================================================

def print_class_information():

    print()
    print("=" * 60)
    print("UNIFIED RETINAL CLASS MAPPING")
    print("=" * 60)

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"{index}: {name}"
        )

    print("=" * 60)


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TESTING RETINAL LABEL MAPPING")
    print("=" * 60)

    print_class_information()

    # --------------------------------------------------------
    # GLOBAL LABEL TEST
    # --------------------------------------------------------

    print()
    print("GLOBAL LABEL TESTS")
    print("-" * 60)

    global_tests = [
        "Diabetic Retinopathy",
        "Glaucoma",
        "AMD",
        "Cataract",
        "Normal"
    ]

    for label in global_tests:

        index = get_class_index(label)

        print(
            f"{label:<25} -> "
            f"{index}"
        )

    # --------------------------------------------------------
    # GAUSSIAN TEST
    # --------------------------------------------------------

    print()
    print("GAUSSIAN TEST")
    print("-" * 60)

    gaussian_labels = [
        "No_DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferate_DR"
    ]

    for label in gaussian_labels:

        index = map_gaussian_label(label)

        print(
            f"{label:<20} -> "
            f"{index} "
            f"({get_class_name(index) if index != UNKNOWN else 'UNKNOWN'})"
        )

    # --------------------------------------------------------
    # OCT TEST
    # --------------------------------------------------------

    print()
    print("OCT TEST")
    print("-" * 60)

    oct_labels = [
        "AMD",
        "CNV",
        "DR",
        "DME",
        "CSR",
        "DRUSEN",
        "MH"
    ]

    for label in oct_labels:

        index = map_oct_label(label)

        name = (
            get_class_name(index)
            if index != UNKNOWN
            else "UNKNOWN"
        )

        print(
            f"{label:<15} -> "
            f"{index:<3} "
            f"({name})"
        )

    # --------------------------------------------------------
    # ODIR TEST
    # --------------------------------------------------------

    print()
    print("ODIR-5K TEST")
    print("-" * 60)

    odir_labels = [
        "ageDegeneration",
        "cataract",
        "diabetes",
        "glaucoma",
        "hypertension",
        "myopia",
        "normal"
    ]

    for label in odir_labels:

        index = map_odir_label(label)

        name = (
            get_class_name(index)
            if index != UNKNOWN
            else "UNKNOWN"
        )

        print(
            f"{label:<20} -> "
            f"{index:<3} "
            f"({name})"
        )

    # --------------------------------------------------------
    # SPLIT TEST
    # --------------------------------------------------------

    print()
    print("SPLIT DATASET TEST")
    print("-" * 60)

    split_labels = [
        "cataract",
        "diabetic retinopathy",
        "glaucoma",
        "normal"
    ]

    for label in split_labels:

        index = map_split_label(label)

        print(
            f"{label:<25} -> "
            f"{index} "
            f"({get_class_name(index)})"
        )

    # --------------------------------------------------------
    # RFMiD TEST
    # --------------------------------------------------------

    print()
    print("RFMiD TEST")
    print("-" * 60)

    rfmid_labels = [
        "DR",
        "Glaucoma",
        "ARMD",
        "Cataract",
        "Normal",
        "Abnormal"
    ]

    for label in rfmid_labels:

        index = map_rfmID_label(label)

        name = (
            get_class_name(index)
            if index != UNKNOWN
            else "UNKNOWN"
        )

        print(
            f"{label:<15} -> "
            f"{index:<3} "
            f"({name})"
        )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("LABEL MAPPING TEST COMPLETE")
    print("=" * 60)

    print()
    print("Final class mapping:")

    for index, name in enumerate(CLASS_NAMES):

        print(
            f"  {index} -> {name}"
        )

    print()
    print("NUM_CLASSES =", NUM_CLASSES)

    print()
    print("IMPORTANT:")
    print(
        "Unsupported/ambiguous labels return -1 "
        "and should be excluded from training."
    )

    print()
    print("=" * 60)