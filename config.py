import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
REFERENCE_IMAGE_DIR = os.path.join(BASE_DIR, "reference_images")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

ROI_DIR = os.path.join(OUTPUT_DIR, "roi")
MASK_DIR = os.path.join(OUTPUT_DIR, "masks")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")

os.makedirs(ROI_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

REFERENCE_IMAGE_DIR = os.path.join(
    BASE_DIR,
    "reference_images_small"
)
ATTUNET_MODEL = os.path.join(
    MODEL_DIR,
    "best_attunet_ham10000.pth"
)

HYBRID_MODEL = os.path.join(
    MODEL_DIR,
    "best_hybrid_cbm.pth"
)

MEMORY_BANK = os.path.join(
    MEMORY_DIR,
    "CARE_MEMORY_BANK_TRAIN.pkl"
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
# CONCEPT NAMES
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
