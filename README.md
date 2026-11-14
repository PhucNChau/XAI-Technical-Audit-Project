# Project 2: Technical Audit of Healthcare Eligibility Scoring Models

> **Target Audience:** Undergraduate Student Teams (Group 2)  
> **Evaluation Mode:** 4-Week Technical Audit, Report, and Oral Defense  
> **Provided Data:** `data/audit_sample.parquet` (10,000 real US Census records across NY and PA)  
> **Model Access:** `vendor_api` black-box scoring service

---

## Environment Setup (Using `uv`)

This course uses **`uv`** as the standard package and environment manager. It creates isolated environments and installs dependencies in 2–3 seconds.

### 1. Install `uv` (One-time setup if needed)
* **macOS / Linux**:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Windows (PowerShell)**:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
  ```

### 2. Create and Activate Your Environment
From inside this project folder:
```bash
# Create isolated environment
uv venv

# Activate the environment
source .venv/bin/activate       # On macOS / Linux
# .venv\Scripts\activate        # On Windows

# Install project dependencies in ~2 seconds
uv pip install -r requirements.txt
```

### 2. Verify Vendor API Connectivity
```python
import pandas as pd
import vendor_api

df_audit = pd.read_parquet("data/audit_sample.parquet")
probs = vendor_api.predict_proba(df_audit)
print("Vendor API successfully initialized. Sample shape:", probs.shape)
```

---

## 4-Week Milestone Timeline

* **Week 1 (Exploration & Baseline)**: Ingest `audit_sample.parquet`. Inspect coverage prevalence and feature distributions. Establish baseline performance metrics on provided validation subsets.
* **Week 2 (Boundary Characterization)**: Sweep the Age feature across the interval $[55, 75]$ to map how the black-box model approximates the statutory age-65 boundary.
* **Week 3 (Perturbation Audit & Surrogate Derivation)**: Quantify the proportion of synthetic neighborhood perturbations that cross the age-65 cliff for a 64-year-old applicant. Measure the degradation in surrogate goodness-of-fit ($R^2$).
* **Week 4 (Cross-State Evaluation & Defense)**: Compare model predictions between NY and PA cohorts. Formulate operational deployment safeguards and finalize your presentation deck.

---

## Final Deliverables

1. **Completed Computational Report**: A fully executed Quarto document (`project_brief.html`) containing all code, derivations, figures, and written interpretations.
2. **Team Presentation & Defense**: A 15-minute presentation defending your findings before the class and responding to cross-examination from peer review teams.
3. **Peer Review Evaluation**: Completed peer review sheet evaluating an assigned peer group's presentation.

