<p align="center">
  <img src="assets/logo.png" width="220">
</p>

<h1 align="center">CARE-MD</h1>

<p align="center">
Clinical Reasoning-Aligned Framework for Self-Explainable Medical Diagnosis
</p>

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Streamlit](https://img.shields.io/badge/Streamlit-WebApp-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

</p>

---

# Framework Architecture

<p align="center">
<img src="assets/architecture.png" width="1000">
</p>

---

# Overview

CARE-MD (Clinical Reasoning-Aligned Framework for Self-Explainable Medical Diagnosis) is an explainable AI framework for dermoscopic skin lesion diagnosis. The framework integrates lesion localization, concept reasoning, episodic memory retrieval, and evidence-based validation to produce transparent and clinically interpretable diagnostic reports.

---

# Key Features

* Automatic ROI Localization using Attention U-Net
* Clinical Concept Reasoning using Hybrid Concept Bottleneck Model
* CARE-Reference Episodic Memory Retrieval
* Top-K Similar Clinical Case Retrieval
* Evidence-based Clinical Decision Support
* Automated Clinical Report Generation
* Explainable AI for Medical Diagnosis

---

# Pipeline

Input Image

↓

Phase-1 : ROI Localization

↓

Phase-2 : Concept Reasoning

↓

Phase-3 : CARE-Reference Retrieval

↓

Phase-4 : Clinical Decision Report
