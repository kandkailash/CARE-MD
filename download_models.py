import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

MODELS = [
    "best_attunet_ham10000.pth",
    "best_hybrid_cbm.pth"
]

for model in MODELS:

    local_path = os.path.join(MODEL_DIR, model)

    if os.path.exists(local_path):
        print(f"✓ {model} already exists")
        continue

    print(f"Downloading {model}...")

    hf_hub_download(
        repo_id="kailashkandpal/CARE-MD-Models",
        filename=model,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )

    print(f"✓ Downloaded {model}")

print("\nAll models are ready.")
