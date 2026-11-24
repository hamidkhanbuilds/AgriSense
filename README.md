# 🌱 AgriSense — An Intelligent Crop Recommendation System

Soil-and-Climate-Based Crop Selection Using Classical Machine Learning

AgriSense takes a farmer's soil nutrient readings (Nitrogen, Phosphorus, Potassium),
environmental conditions (temperature, humidity, rainfall) and soil pH, and recommends
the single most suitable crop out of 22 common crops. It is trained on a public labeled
dataset and deployed as a simple Streamlit web app.

---

## Objectives

- Help farmers choose the right crop for their field using data instead of guesswork.
- Train and compare four classical machine learning classifiers.
- Select the best model by experimental results and deploy it in a web app.

---

## Technologies and Libraries

| Category | Tools |
|---|---|
| Language | Python 3.x |
| ML / Modelling | scikit-learn (RandomForest, DecisionTree, KNN, GaussianNB) |
| Data Handling | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Version Control | Git, GitHub |

---

## Project Structure

```
AgriSense/
├── data/
│   └── Crop_recommendation.csv      # Dataset (2200 rows, 22 crops)
├── src/
│   ├── preprocessing.py             # Load, clean, encode, scale, split
│   ├── eda.py                       # Exploratory data analysis plots
│   ├── training.py                  # Train the 4 models
│   ├── evaluation.py                # Metrics, confusion matrices, comparison
│   └── main.py                      # Runs the full pipeline in order
├── results/                         # Plots, metrics, reports (generated)
├── app.py                           # Streamlit web application
├── requirements.txt
├── README.md
└── AgriSense_Explanation.docx       # Detailed explanation document
```

---

## Dataset

- **Name:** Crop Recommendation Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset
- **Size:** 2,200 records, 7 input features, 22 crop classes (100 samples each)
- **Input features:** N, P, K, temperature, humidity, pH, rainfall
- **Target:** crop label (multi-class classification)

The dataset is already clean (0 missing values, 0 duplicates, perfectly balanced).
The preprocessing code still runs the missing-value and duplicate checks so the pipeline
is safe on any similar data.

---

## Installation

1. Make sure Python 3.9 or newer is installed.
2. (Optional) create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
3. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

**Step 1 — Generate the results and plots for the report:**
```bash
python src/main.py
```
This loads the data, runs EDA, trains all four models, evaluates them, and saves
every plot and the metrics table to `results/`.

**Step 2 — Launch the web app:**
```bash
streamlit run app.py
```
Open the browser link shown in the terminal, enter your field readings, and click
**Recommend Crop**. The app trains the Random Forest model on startup from the dataset,
so it works on its own — you do not need to run Step 1 first.

---

## Model Information

Four classical classifiers are trained on the same 80% train / 20% test split:

1. **Random Forest** — ensemble of decision trees (n_estimators=200).
2. **Decision Tree** — single Gini-based tree, interpretable baseline.
3. **KNN** — k-nearest neighbours (k=5) on scaled features.
4. **Naive Bayes** — GaussianNB probabilistic classifier.

The best model is chosen automatically by the highest macro **F1-score** (Random Forest).
The web app trains this same model on startup and uses it for predictions.

---

## Results Summary

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| **Random Forest** | **0.9955** | **0.9957** | **0.9955** | **0.9955** |
| Naive Bayes | 0.9955 | 0.9959 | 0.9955 | 0.9954 |
| Decision Tree | 0.9795 | 0.9806 | 0.9795 | 0.9794 |
| KNN | 0.9795 | 0.9804 | 0.9795 | 0.9793 |

**Best model: Random Forest** — highest macro F1-score, only 2 misclassifications out
of 440 test samples, and it provides feature-importance scores. (Exact values may vary
by a fraction depending on your scikit-learn version.)

---

## Group Members

| Name | Roll Number |
|---|---|
| Hamid Khan | 23P-0044 |
| Ahmad Jan | 23P3007 |

**Hamid Khan** — Data preprocessing, EDA, Random Forest and Decision Tree training,
model comparison report, GitHub setup and README.

**Ahmad Jan** — KNN and Naive Bayes training, evaluation metrics and comparison,
Streamlit web app development, project report writing.

---

## Submission Deliverables

- Source code (this repository / ZIP)
- Project report (`AgriSense_Report.docx`)
- Dataset: Kaggle link above (also included in `data/`)
- Web app demo: `streamlit run app.py`
- Live web app Link: `https://agrisense-crop.streamlit.app`
