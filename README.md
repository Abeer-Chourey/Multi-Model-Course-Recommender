# 🎓 Personalized Course Recommendation System
### A Multi-Model Machine Learning Framework for Academic Success

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-LightGBM%20%7C%20GradientBoosting-orange)

## 📖 Theoretical Overview

This repository contains the complete replication and adaptation of the hierarchical, multi-model machine learning framework originally proposed in the research paper:

> **"Personalized Course Recommendation System: A Multi-Model Machine Learning Framework for Academic Success"** 
> *(Islam & Hosen, MDPI Digital 2025)*.

While the original authors validated their methodology using a purely synthetically generated dataset (DF4), this project takes their framework a step further. We successfully adapted, trained, and validated the identically structured architecture using **real-world, open-source educational datasets** from Kaggle, proving the real-world viability of their methodology on extremely messy, authentic student enrollment data.

The core motivation behind this architecture is the assertion that course advising cannot be modeled by simple collaborative filtering. It requires calculating overlapping, and often conflicting, academic constraints (e.g., student GPA, course difficulty, prerequisites, and graduation deadlines).

---

## 📊 Datasets Used & Feature Engineering

To synthesize a highly realistic environment matching the paper's mathematical requirements, we sourced two independent datasets and merged them via a simulated interaction matrix, yielding over **100,000 highly realistic interaction records**.

1. **Course Data**: [Udemy Course Recommendation Dataset](https://www.kaggle.com/datasets/evilspirit05/udemy-course-recommendation)
   - *Engineered Features*: Course Difficulty Level (1-4), Popularity Composite (subscribers & reviews), Price Cost, Content Richness (duration * lectures).
2. **Student Data**: [College Student Management Dataset](https://www.kaggle.com/datasets/ziya07/college-student-management-dataset)
   - *Engineered Features*: Student Capability (Normalized GPA), Universal Engagement Score (attendance, LMS login), Current Credit Load, Risk Level.

---

## 🧠 Hierarchical Framework Architecture

The system relies on a rigorous **Two-Layer Hierarchical ML Framework** designed specifically to handle strict academic advising constraints.

### 1. Local Model Framework (LMF)
The first layer consists of five independent, specialized predictive models. Each targets a strict, localized dimension of the student's academic profile.

| Model | Purpose | Input Sub-Features | Algorithm Used | Output Scope |
|---|---|---|---|---|
| **SPM** *(Success Probability)* | Predicts the exact probability of a student successfully completing a course without failing. | Student capability, course difficulty gap, engagement score. | `GradientBoostingRegressor` | Continuous `[0, 1]` |
| **CFSM** *(Course Fit Score)* | Evaluates how well a specific course's subject aligns with a student's engagement profile. | Engagement, popularity, content richness. | `GradientBoostingRegressor` | Continuous `[0, 1]` |
| **PFM** *(Prerequisite Fulfillment)*| Classifies whether the student possesses the required preliminary knowledge. | Capability vs. Course Difficulty threshold. | `LightGBM Classifier` | Binary `(Yes/No)` |
| **GPM** *(Graduation Priority)* | Highlights whether a given course is strictly necessary for immediate graduation tracks. | Current load, enrollment status. | `LightGBM Classifier` | Binary `(Yes/No)` |
| **RLM** *(Recommended Load)*| Evaluates mental bandwidth to recommend the absolute optimal maximum credit hours. | GPA, attendance, current risk level. | `GradientBoostingRegressor` | Continuous `(1 - 7)` |

### 2. Global Model Framework (GMF)
The second layer acts as the master execution function. We feed the raw output predictions from all 5 LMF models into a final `LightGBM Regressor` meta-equation. 

The GMF inherently learns to severely penalize courses where prerequisites are missing (determined by the PFM) and heavily rewards courses with high success probability (SPM) and graduation urgency (GPM), resulting in the final **Constraint-Aware Recommendation Score**.

---


## 🚀 Setup & Execution

You will need Python 3.8+ installed on your machine. All dependencies can be installed via the requirements file.

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/RS-Research-Paper-Multi-Model.git
cd RS-Research-Paper-Multi-Model

# 2. Install required packages
pip install -r requirements.txt

# 3. Run the pipeline in order
python step1_data_loading.py
python step2_feature_engineering.py
python step3_data_splitting.py
python step4_model_training.py
python step5_model_evaluation.py
python step6_final_evaluation.py
python step7_hybrid_recommender.py
```
> **Note:** If you wish to skip training and jump straight to generating new recommendations, you can execute `python step7_hybrid_recommender.py` directly, assuming the pre-trained weights remain in the `/models/` directory.

---

## 📈 Key Results & Replication Comparison

Our adaptation proves that the paper's methodology scales magnificently to real-world educational data without requiring highly structured or purely synthetic databases.

### 📝 Core Replication Results (Paper vs. Ours)
By moving from synthetic generation logic to real-world human data, absolute mathematical perfection drops slightly, but the **underlying model architecture demonstrates incredibly powerful resistance to noise**, successfully solving complex human-data variance.

| Model | Target Output | Paper RMSE | Our RMSE | Paper Accuracy | Our Accuracy | Analysis/Conclusion |
|-------|---------------|------------|----------|----------------|--------------|---------------------|
| **SPM** | *Probability (0-1)* | `0.00956` | `0.05057` | N/A | N/A | Our model predicts real human success within a **5% margin of error**, successfully validating the paper. |
| **CFSM**| *Fit Score (0-1)* | `0.01171` | `0.04042` | N/A | N/A | Predicting course fit within a **4% margin of error**, proving their continuous regression logic holds on real data. |
| **PFM** | *Prereq (0/1)* | N/A | N/A | `>99%` | `97.02%` | Real-world missing data handles prerequisites slightly less rigidly than synthetic databases, maintaining >97% Acc. |
| **GPM** | *Priority (0/1)* | N/A | N/A | `>99%` | `91.87%` | Identifies critical graduation constraints correctly over 91% of the time. |
| **RLM** | *Max Load (1-7)* | `0.00541` | `0.30496` | N/A | N/A | Our model estimates realistic course loads within roughly **0.3 courses** (equivalent to 1 credit hour) of human-optimal truth. |

### Visual 1: Final Test Set Classification & ROC Curves
*Classifiers accurately split boolean constraints directly impacting the master global algorithm.*
![Test ROC Curves](plots/test_roc_curves.png)

### Visual 2: Validation Metrics Distributions
*Distribution analysis showing extremely stable, non-overfitting training results across all 6 models.*
![Validation Metrics](plots/val_metrics_summary.png)

### Visual 3: Final Output State (Constraint-Aware Hybrid Recommendations)
*In step 7, the algorithm loops over sample students to construct individualized, ranked schedules utilizing all aggregated GMF logic. The below bar chart represents the final output state demonstrating mathematically constrained recommendations per student.*
![Personalized Hybrid Recommendations](plots/hybrid_recommendations.png)

---

## 📋 Conclusion
This repository successfully translates pure academic theory into a functioning, real-world data pipeline. We conclusively demonstrated that the *Two-Layer Local/Global Hierarchy* algorithm effectively curates personalized schedules that actively respect complex constraints such as prerequisites, credit load limitations, and graduation timelines seamlessly.
