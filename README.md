# 🌾 Crop Recommendations with Ensemble Learning and Explainable AI

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-green.svg)](https://shap.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A climate-aware, interpretable crop recommendation system combining ensemble 
> machine learning with SHAP-based Explainable AI — deployed as a Streamlit web application.

---

## 📌 Project Overview

Most existing crop recommendation systems suffer from two fundamental limitations:
- Trained on small, single-source datasets (≤2,200 samples) leading to overfitting
- Black-box predictions with no explanation of why a crop was recommended

This project addresses both limitations through:

| Contribution | Description |
|---|---|
| **Multi-source dataset** | 3,463 clean samples merged from 4 independent Kaggle repositories |
| **Novel CSI feature** | Climate Stress Index encoding composite climate deviation |
| **Rigorous evaluation** | 5-fold stratified CV with confidence intervals for all 5 models |
| **3-level SHAP XAI** | Global + per-class + instance-level explanations |
| **Streamlit deployment** | Web app with real-time SHAP waterfall explanations |

---

## 📊 Results

| Model | CV Accuracy | Std Dev | 95% CI |
|---|---|---|---|
| Decision Tree | 85.20% | ±0.61% | 84.0%–86.4% |
| Random Forest | 88.13% | ±0.80% | 86.6%–89.7% |
| XGBoost | 87.38% | ±0.57% | 86.3%–88.5% |
| Ensemble (RF+DT+XGB) | 86.55% | ±0.66% | 85.3%–87.8% |
| **Tuned RF (proposed)** | **88.25%** | **±0.51%** | **87.25%–89.25%** |

✅ 18 of 26 crop classes achieve F1-score ≥ 0.96

---

## 🌱 Supported Crops (26 Classes)
apple, banana, barley, blackgram, chickpea, coconut, coffee, cotton,
grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean,
muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, soybean,
sugarcane, watermelon, wheat


---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/crop-recommendation-xai.git
cd crop-recommendation-xai
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
Download these 4 datasets from Kaggle and place in `data/raw/`:

| Save as | Kaggle URL |
|---|---|
| `crop_data.csv` | kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| `crop_data_v2.csv` | kaggle.com/datasets/atharvaingle/crop-recommendation-dataset |
| `crop_data_v3.csv` | kaggle.com/datasets/aksahaha/crop-recommendation |
| `crop_data_v4.csv` | kaggle.com/datasets/miadul/crop-recommendation-dataset |

### 5. Run the notebooks
```bash
jupyter notebook
```
Open and run `research/notebooks/02_expanded_model.ipynb` top to bottom.
This generates the model bundle at `models/model_bundle.pkl`.

### 6. Launch the Streamlit app
```bash
venv\Scripts\python.exe -m streamlit run app/app.py
```
Open browser at: **http://localhost:8501**

---

## 📁 Project Structure
crop-recommendation-xai/
│
├── README.md                   # This file
├── requirements.txt            # All dependencies
├── .gitignore                  # Files excluded from git
│
├── data/
│   ├── raw/                    # Place downloaded CSV files here
│   │   ├── crop_data.csv
│   │   ├── crop_data_v2.csv
│   │   ├── crop_data_v3.csv
│   │   └── crop_data_v4.csv
│   └── processed/              # Auto-generated: clean merged dataset
│       ├── crop_merged_clean.csv
│       └── expanded_data.pkl
│
├── research/
│   └── notebooks/
│       ├── 01_baseline_setup.ipynb     # Original baseline models
│       └── 02_expanded_model.ipynb     # Full pipeline notebook
│
├── app/
│   └── app.py                  # Streamlit web application
│
├── models/
│   └── model_bundle.pkl        # Auto-generated: trained model + encoder
│
└── results/
├── figures/                # Auto-generated: all plots and charts
└── tables/                 # Auto-generated: CSV result tables

---

## 🔬 Methodology

### Dataset Construction
- Merged 4 independent Kaggle datasets (7,600 raw rows)
- Removed 3,674 duplicate records and 463 outliers
- Final clean dataset: **3,463 samples, 26 crop classes**

### Feature Engineering (7 → 22 features)
| Category | Features | Count |
|---|---|---|
| Base agronomic | N, P, K, temperature, humidity, ph, rainfall | 7 |
| Nutrient ratios | N/P, N/K, P/K, NPK_total, NPK_balance | 5 |
| Climate interactions | temp×humidity, rainfall×humidity, temp×rainfall | 3 |
| Climate Stress Index | CSI = 0.4\|T̂\| + 0.3\|Ĥ\| + 0.3\|R̂\| | 1 |
| pH bands | ph_acidic, ph_neutral, ph_alkaline | 3 |
| Rainfall bands | rainfall_low, rainfall_medium, rainfall_high | 3 |

### Climate Stress Index (CSI)

scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.43.0
imbalanced-learn>=0.11.0
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0

---

## 🎯 Alignment with SDGs

| SDG | Contribution |
|---|---|
| SDG 2 — Zero Hunger | Improves crop-soil matching for smallholder farmers |
| SDG 9 — Innovation | Novel CSI feature and 3-level SHAP framework |
| SDG 12 — Responsible Consumption | Reduces unnecessary fertiliser and water use |
| SDG 13 — Climate Action | Climate-aware CSI addresses agricultural adaptation |

---

## 👥 Authors

| Name | Role | University |
|---|---|---|
| Ashutosh Agrawal | Data Engineer & Lead Developer | Presidency University, Bengaluru |
| Anamaneni Abhilash | ML Engineer | Presidency University, Bengaluru |
| Ashutosh Kumar Sharma | XAI Specialist & Technical Writer | Presidency University, Bengaluru |

**Supervisor:** Dr. Sandhya L., Assistant Professor,  
School of Computer Science and Engineering, Presidency University, Bengaluru

---

## 📄 Citation

If you use this work, please cite:
```bibtex
@inproceedings{agrawal2025crop,
  title     = {Crop Recommendations with Ensemble Learning and Explainable AI},
  author    = {Agrawal, Ashutosh and Abhilash, Anamaneni and Sharma, Ashutosh Kumar},
  booktitle = {Proceedings of the IEEE International Conference on Computational
               Intelligence and Data Science},
  year      = {2025},
  publisher = {IEEE}
}
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- Dr. Sandhya L. — project supervisor and guidance
- Kaggle contributors for the public agricultural datasets
- Open-source community behind scikit-learn, SHAP, XGBoost, and Streamlit

scikit-learn>=1.3.0
xgboost>=1.7.0
shap>=0.43.0
imbalanced-learn>=0.11.0
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
notebook>=6.5.0
ipykernel>=6.0.0
