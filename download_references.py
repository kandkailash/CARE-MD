import os
import zipfile
from huggingface_hub import hf_hub_download

REFERENCE_DIR = "reference_images_small"
ZIP_FILE = "reference_images_small.zip"


def ensure_reference_images():

    # Already extracted
    if os.path.isdir(REFERENCE_DIR):
        print("✓ Reference images already available")
        return

    print("Downloading reference images...")

    zip_path = hf_hub_download(
        repo_id="kailashkandpal/CARE-MD-Reference-Images",
        repo_type="dataset",
        filename=ZIP_FILE,
        local_dir="."
    )

    print("Extracting images...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(".")

    print("✓ Reference images ready")
