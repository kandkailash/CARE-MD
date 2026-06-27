import os

# ==========================================================
# ROOT
# ==========================================================

ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

# ==========================================================
# MODELS
# ==========================================================

ATTUNET_MODEL = os.path.join(
    ROOT,
    "models",
    "best_attunet_ham10000.pth"
)

HYBRID_MODEL = os.path.join(
    ROOT,
    "models",
    "best_hybrid_cbm.pth"
)

# ==========================================================
# MEMORY
# ==========================================================

MEMORY_BANK = os.path.join(
    ROOT,
    "memory",
    "CARE_MEMORY_BANK_TRAIN.pkl"
)

# ==========================================================
# REFERENCE IMAGES
# ==========================================================

REFERENCE_IMAGE_DIR = os.path.join(
    ROOT,
    "data",
    "reference_images"
)

# ==========================================================
# OUTPUTS
# ==========================================================

OUTPUT_DIR = os.path.join(
    ROOT,
    "outputs"
)

ROI_DIR = os.path.join(
    OUTPUT_DIR,
    "roi"
)

MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

REPORT_DIR = os.path.join(
    OUTPUT_DIR,
    "reports"
)

UPLOAD_DIR = os.path.join(
    ROOT,
    "uploads"
)

# ==========================================================
# CREATE DIRECTORIES
# ==========================================================

for path in [

    OUTPUT_DIR,

    ROI_DIR,

    MASK_DIR,

    REPORT_DIR,

    UPLOAD_DIR

]:

    os.makedirs(
        path,
        exist_ok=True
    )

# ==========================================================
# CLASS NAMES
# ==========================================================

CLASS_NAMES = [

    "akiec",

    "bcc",

    "bkl",

    "df",

    "mel",

    "nv",

    "vasc"

]

# ==========================================================
# CLINICAL CONCEPTS
# ==========================================================

CONCEPT_NAMES = [

    "Asymmetry",

    "Border",

    "Color",

    "Diameter",

    "Compactness",

    "Color Entropy",

    "Latent-1",

    "Latent-2",

    "Latent-3"

]