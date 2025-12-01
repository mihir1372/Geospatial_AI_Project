# Explainability and Fairness in FireRisk Classification  
### Analyzing Low Accuracy in Remote Sensing Using XAI and Fairness Techniques  
**Author: Mihir Hemal Shah**

This repository presents a study on improving the interpretability and fairness of FireRisk classification models.  
Our goal is to understand the root causes behind the relatively low accuracy in FireRisk prediction and apply both **Explainable AI (XAI)** and **fairness-aware algorithms** to analyze model behavior, dataset imbalance, and prediction bias.

---

## General Configuration (Kaggle Setup)

To reproduce results using Kaggle:

### **1. Upload Notebook**
Start by uploading the required experiment notebooks to a Kaggle notebook environment.

### **2. Dataset**
Use the FireRisk dataset uploaded on Kaggle:
- **Full FireRisk All Samples** dataset

### **3. Checkpoint**
Load the trained model checkpoint available on Kaggle:
- **FireRisk Model Checkpoint**

### **4. Accelerator Settings**
Under **Accelerator**, select:
- **GPU X T4 (Dual GPU)**  
This is required because loading the full 16GB dataset under a single GPU can lead to memory issues.

### **5. Working With Subsets**
For faster experimentation or initial runs, use the **10% sample dataset**:
- *FireRisk Classification Sample Data*

---

## Folder Structure

### **Baseline/**
Contains:
- `full_dataset_fire_risk_baseline.ipynb`  
Run this notebook using the full dataset.  
Follow the configuration setup and execute all cells up to the confusion matrix and classification report section.

---

### **Experiment_S1_S2_S3/**
Includes three experiment notebooks:

- `full-dataset-fire-risk_s1.ipynb`  
- `full-dataset-fire-risk_s2.ipynb`  
- `full-dataset-fire-risk_s3.ipynb`

These correspond to the hybrid EuroSAT + FireRisk modeling approaches:
- **S1:** Full fine-tuning  
- **S2:** Fine-tuning only the classification head  
- **S3:** Concatenated classification heads (best result)

Run each notebook fully to view accuracy, loss curves, and comparative results.

---

### **XAI/**
Contains:
- `full_dataset_fire_risk_xai_tsne.ipynb`

This notebook generates:
- Grad-CAM++ activation heatmaps  
- t-SNE feature visualizations  

**Notes:**
- If GPU RAM runs out during t-SNE, restart the kernel and run only the t-SNE section.  
- The XAI and t-SNE steps are independent; you may skip one if needed.

---

### **Fairness/**
Contains three fairness-related notebooks:

- `final-project-csc-791-geospatial-initial.ipynb` – computes initial fairness metrics (TPR, FPR, EOD)  
- `class-re-weighting.ipynb` – applies class-weight adjustments  
- `fairlearn-mitigator.ipynb` – applies Fairlearn’s mitigation strategies  

Run these notebooks in order to evaluate fairness baselines, adjusted models, and mitigated results.

---
