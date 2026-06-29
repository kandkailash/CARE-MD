import os
import zipfile
from huggingface_hub import hf_hub_download

REFERENCE_DIR = "reference_images_small"
ZIP_FILE = "reference_images_small.zip"


def ensure_reference_images():

    # Already downloaded & extracted
    if (
        os.path.exists(REFERENCE_DIR)
        and len(os.listdir(REFERENCE_DIR)) > 7000
    ):
        print("✓ Reference images already available")
        return

    print("Downloading reference images...")

    zip_path = hf_hub_download(
        repo_id="kailashkandpal/CARE-MD-Reference-Images",
        repo_type="dataset",
        filename=ZIP_FILE,
        local_dir="."
    )

    print("Extracting reference images...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    print("✓ Reference images extracted")

    # Optional: delete ZIP after extraction
    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("✓ CARE-MD reference images ready")

    print("Reference folder:", REFERENCE_DIR)
    print("Files extracted:", len(os.listdir(REFERENCE_DIR)))
    print("First files:", os.listdir(REFERENCE_DIR)[:5])
