# CARE-MD

## Clinical Reasoning-Aligned Framework for Self-eXplainable Medical Diagnosis

CARE-MD is a multi-phase explainable AI framework for skin lesion diagnosis. The framework combines lesion localization, concept-based reasoning, case-based retrieval, and evidence-driven clinical reporting to provide transparent and interpretable predictions.

---

# Features

- Automatic Lesion Localization (Attention U-Net)
- Clinical Concept Prediction (Hybrid CBM)
- Episodic Memory Retrieval
- Similar Case Retrieval
- Evidence-based Diagnosis
- Clinical AI Explanation
- Automated Clinical Report Generation

---

# Framework

Input Image

↓

Phase-1
ROI Localization
(Attention U-Net)

↓

Phase-2
Concept Reasoning
(Hybrid Concept Bottleneck Model)

↓

Phase-3
CARE-Reference
(Episodic Memory Retrieval)

↓

Phase-4
Clinical Decision Report

---

# Repository Structure

```
CARE-MD/
│
├── app.py
├── caremd.py
├── config.py
├── requirements.txt
├── README.md
│
├── models/
├── memory/
├── reference_images/
├── uploads/
├── outputs/
├── assets/
└── examples/
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/CARE-MD.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
streamlit run app.py
```

---

# Pipeline

1. Upload Image
2. ROI Extraction
3. Concept Prediction
4. Similar Case Retrieval
5. Evidence Validation
6. Clinical Report Generation

---

# Output

The generated report contains

- Original Image
- ROI
- Diagnosis Summary
- Clinical Concepts
- Similar Clinical References
- Evidence Summary
- Clinical AI Explanation

---

# Models

The trained model weights are **not included** in this repository because of GitHub file size limitations.

Download the pretrained models and place them inside the **models/** directory.

---

# Citation

If you use CARE-MD in your research, please cite the corresponding publication.

---

# License

MIT License
