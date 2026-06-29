import os
import zipfile
from huggingface_hub import hf_hub_download

REFERENCE_DIR = "reference_images_small"
ZIP_FILE = "reference_images_small.zip"


def ensure_reference_images():

    if os.path.exists(REFERENCE_DIR):
        print("✓ Reference images already available")
        return

    os.makedirs(REFERENCE_DIR, exist_ok=True)

    print("Downloading reference images...")

    zip_path = hf_hub_download(
        repo_id="kailashkandpal/CARE-MD-Reference-Images",
        repo_type="dataset",
        filename=ZIP_FILE
    )

    print("ZIP Path:", zip_path)

    print("Extracting...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(REFERENCE_DIR)

    print("Files:", len(os.listdir(REFERENCE_DIR)))
    print("Done.")
